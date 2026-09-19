first_move_modifier=0.1
eigenvalue_cutoff = -0.2
move_phase1_modifier=0.1
move_phase2_modifier = 0.001
#force_cutoff_saddlepoint = 1.5
force_on_atom_cutoff_saddlepoint = 0.08
dot_product_saddle_cutoff = 0.00025

#=================================================================#
#Relaxation parameters

relaxation_step_magnitude=0.0005
max_relaxation_steps = 50
relaxation_step_magnitude_multiplier_success = 1.05
relaxation_step_magnitude_multiplier_failure = 0.5
#inpath2 = "C:\\Users\\Yanik\\Desktop\\Master Thesis\\config_intermediate.lammps"
inpath2 = "//home//schwyyzer//Desktop//Master Thesis//config_small_relaxed.lammps"
#csvfile = "C:\\Users\\Yanik\\Desktop\\Master Thesis\\results_eigenvectors(in).csv"
csvfile = nothing

mass = [2 1]
epsilon_table = [1 1; 1 1]
sigma_table = [1 11/12; 11/12 5/6]
rc = 6 # LJ cutoff
