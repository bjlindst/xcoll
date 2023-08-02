// copyright ############################### #
// This file is part of the Xcoll Package.   #
// Copyright (c) CERN, 2023.                 #
// ######################################### #

#ifndef XCOLL_EVEREST_H
#define XCOLL_EVEREST_H
#include <math.h>
#include <stdio.h>


/*gpufun*/
void EverestCollimator_set_material(EverestCollimatorData el, LocalParticle* part0){
    MaterialData material = EverestCollimatorData_getp__material(el);
    RandomRutherfordData rng = EverestCollimatorData_getp_rutherford_rng(el);
    RandomRutherford_set_by_xcoll_material(rng, (GeneralMaterialData) material);
}


// TODO: it would be great if we could set EverestData as an xofield, because then we could
// run this function at creation of the collimator instead of every turn
/*gpufun*/
EverestData EverestCollimator_init(EverestCollimatorData el, LocalParticle* part0){

    EverestData coll = (EverestData) malloc(sizeof(EverestData_));

    // Random generator and material
    coll->rng = EverestCollimatorData_getp_rutherford_rng(el);
    MaterialData material = EverestCollimatorData_getp__material(el);
    coll->exenergy = MaterialData_get_excitation_energy(material)*1.0e3; // MeV
    coll->rho      = MaterialData_get_density(material);
    coll->anuc     = MaterialData_get_A(material);
    coll->zatom    = MaterialData_get_Z(material);
    coll->bnref    = MaterialData_get_nuclear_elastic_slope(material);
    coll->radl     = MaterialData_get_radiation_length(material);
    coll->csref[0] = MaterialData_get_cross_section(material, 0);
    coll->csref[1] = MaterialData_get_cross_section(material, 1);
    coll->csref[5] = MaterialData_get_cross_section(material, 5);

    // Impact table
    coll->record = EverestCollimatorData_getp_internal_record(el, part0);
    coll->record_index = NULL;
    if (coll->record){
        coll->record_index = CollimatorImpactsData_getp__index(coll->record);
    }

    // Geometry
    // TODO: this should in principle not be in this struct
    coll->aperture = EverestCollimatorData_get_jaw_L(el) - EverestCollimatorData_get_jaw_R(el);
    coll->offset   = ( EverestCollimatorData_get_jaw_L(el) + EverestCollimatorData_get_jaw_R(el) ) /2;
    coll->tilt_L   = asin(EverestCollimatorData_get_sin_yL(el));
    coll->tilt_R   = asin(EverestCollimatorData_get_sin_yR(el));
    coll->side     = EverestCollimatorData_get__side(el);

    return coll;
}


/*gpufun*/
void EverestCollimator_track_local_particle(EverestCollimatorData el, LocalParticle* part0) {
    int8_t active = EverestCollimatorData_get_active(el);
    active       *= EverestCollimatorData_get__tracking(el);
    double const inactive_front = EverestCollimatorData_get_inactive_front(el);
    double const active_length  = EverestCollimatorData_get_active_length(el);
    double const inactive_back  = EverestCollimatorData_get_inactive_back(el);

    // Collimator geometry
    double const co_x       = EverestCollimatorData_get_ref_x(el);
    double const co_y       = EverestCollimatorData_get_ref_y(el);
    // TODO: we are ignoring the angle of the right jaw
    double const sin_zL     = EverestCollimatorData_get_sin_zL(el);
    double const cos_zL     = EverestCollimatorData_get_cos_zL(el);
    double const sin_zR     = EverestCollimatorData_get_sin_zR(el);
    double const cos_zR     = EverestCollimatorData_get_cos_zR(el);
    if (fabs(sin_zL-sin_zR) > 1.e-10 || fabs(cos_zL-cos_zR) > 1.e-10 ){
        kill_all_particles(part0, XC_ERR_NOT_IMPLEMENTED);
    };

    // Initialise collimator data
    // TODO: we want this to happen before tracking (instead of every turn), as a separate kernel
    EverestData coll = EverestCollimator_init(el, part0);

    // Preinitialise scattering parameters
    double const energy0 = LocalParticle_get_energy0(&part0[0]) / 1e9; // Reference energy in GeV
    calculate_scattering(coll, energy0, 1.);
    calculate_ionisation_properties(coll, energy0);

    //start_per_particle_block (part0->part)
        if (!active){
            // Drift full length
            Drift_single_particle(part, inactive_front + active_length + inactive_back);

        } else {
            // Check collimator initialisation
            int8_t is_tracking = assert_tracking(part, XC_ERR_INVALID_TRACK);
            int8_t rng_is_set  = assert_rng_set(part, RNG_ERR_SEEDS_NOT_SET);
            int8_t ruth_is_set = assert_rutherford_set(coll->rng, part, RNG_ERR_RUTH_NOT_SET);

            if (is_tracking && rng_is_set && ruth_is_set) {
                // Drift inactive front
                Drift_single_particle(part, inactive_front);

                // Move to collimator frame
                XYShift_single_particle(part, co_x, co_y);
                SRotation_single_particle(part, sin_zL, cos_zL);

                scatter(coll, part, active_length);

                // Return from collimator frame
                SRotation_single_particle(part, -sin_zL, cos_zL);
                XYShift_single_particle(part, -co_x, -co_y);

                // Drift inactive back (only surviving particles)
                if (LocalParticle_get_state(part) > 0){
                    Drift_single_particle(part, inactive_back);
                }
            }
        }
    //end_per_particle_block

    free(coll);
}


#endif /* XCOLL_EVEREST_H */
