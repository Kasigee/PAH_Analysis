import numpy as np
import pandas as pd
import glob
import multiprocessing as mp
import hashlib
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for environments without a display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, dual_annealing, differential_evolution
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

###############################################################################
# Function to calculate error metrics for modeling approach
###############################################################################
def calculate_mad(coeffs, df, variables, target):
    """
    1) Compute WeightedSum = np.dot(df[variables], coeffs).
    2) OLS regression of WeightedSum -> target.
    3) Return error metrics, regression_slope, regression_intercept, etc.
    """
    # Create a weighted sum of the features using the coefficients
    weighted_sum = np.dot(df[variables], coeffs)

    # Reshape weighted_sum to 2D array for sklearn
    weighted_sum_2d = weighted_sum.reshape(-1, 1)

    # Perform a linear regression to find the best fit line
    model = LinearRegression()
    model.fit(weighted_sum_2d, df[target])

    # Get the regression parameters (slope and intercept)
    regression_slope = model.coef_[0]
    regression_intercept = model.intercept_

    # Use the model to predict the target values
    predicted = model.predict(weighted_sum_2d)

    # Calculate the errors
    errors = predicted - df[target]

    # Calculate MBE (Mean Bias Error)
    mbe = np.mean(errors)

    # Calculate MAD (Mean Absolute Deviation)
    mad = np.mean(np.abs(errors))

    # Calculate MSE (Mean Signed Error)
    mse = np.mean(errors)

    # Calculate RMSD (Root Mean Squared Deviation)
    rmsd = np.sqrt(np.mean(errors ** 2))

    # Calculate min_error and max_error
    min_error = np.min(np.abs(errors))
    max_error = np.max(np.abs(errors))

    # Calculate MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_errors = np.abs(errors / df[target])
        percentage_errors = percentage_errors[np.isfinite(percentage_errors)]
        mape = np.mean(percentage_errors) * 100 if len(percentage_errors) > 0 else np.nan

    # Handle zero targets for MAPD
    valid_target = df[target] != 0
    if valid_target.any():
        mapd = np.mean(np.abs(errors[valid_target] / df[target][valid_target])) * 100
    else:
        mapd = np.nan

    # R-squared
    r_squared = r2_score(df[target], predicted)

    return (
        mad, rmsd, mse, 
        min_error, max_error, 
        mape, mapd, 
        r_squared, mbe, 
        predicted, regression_slope, regression_intercept
    )

###############################################################################
# Function for direct error computation (unchanged from your script)
###############################################################################
def compute_direct_errors(df, variable, target):
    # Direct comparison without fitting a model
    errors = df[variable] - df[target]

    mbe = np.mean(errors)
    mad = np.mean(np.abs(errors))
    mse = np.mean(errors)
    rmsd = np.sqrt(np.mean(errors ** 2))
    min_error = np.min(np.abs(errors))
    max_error = np.max(np.abs(errors))

    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_errors = np.abs(errors / df[target])
        percentage_errors = percentage_errors[np.isfinite(percentage_errors)]
        mape = np.mean(percentage_errors) * 100 if len(percentage_errors) > 0 else np.nan

    valid_target = df[target] != 0
    if valid_target.any():
        mapd = np.mean(np.abs(errors[valid_target] / df[target][valid_target])) * 100
    else:
        mapd = np.nan

    r_squared = r2_score(df[target], df[variable])
    predicted = df[variable].values

    # For direct comparison, regression slope is 1, intercept is 0
    regression_slope = 1.0
    regression_intercept = 0.0

    return (
        mad, rmsd, mse, 
        min_error, max_error, 
        mape, mapd, 
        r_squared, mbe, 
        predicted, regression_slope, regression_intercept
    )

###############################################################################
# Updated minimize_mad function with "Alternate Simplification"
###############################################################################
def minimize_mad(df, variables, target, initial_guess=None,
                 method='Nelder-Mead', scale=True, direct_comparison=False):

    if direct_comparison:
        # Perform direct error computation
        result = compute_direct_errors(df, variables[0], target)
        (mad, rmsd, mse, 
         min_error, max_error, 
         mape, mapd, 
         r_squared, mbe, 
         predicted, 
         regression_slope, 
         regression_intercept) = result

        optimized_coeffs = None
        optimized_coeffs_str = 'N/A'
        # Cross-validation is not applicable for direct comparison
        cv_mad = cv_rmsd = cv_mse = cv_mape = cv_r_squared = cv_mbe = np.nan

        # final_intercept and final_slopes for direct comparison
        final_intercept = regression_intercept
        final_slopes = [regression_slope]  # one variable
        final_slopes_str = ', '.join(f'{fs:.8f}' for fs in final_slopes)

    else:
        if initial_guess is None:
            initial_guess = [1.0] * len(variables)  # Default initial guess

        # Make a copy of df to avoid altering outside data
        df = df.copy()
        scaler = None

        # Optionally scale the variables
        if scale:
            scaler = StandardScaler()
            df[variables] = scaler.fit_transform(df[variables])

        def objective_function(coeffs):
            # Extract only MAD for optimization
            (mad_local, _, _, _, _, _, _, _, _, _, _, _) = calculate_mad(
                coeffs, df, variables, target
            )
            return mad_local

        # Use the chosen method to minimize MAD
        if method == 'differential_evolution':
            bounds = [(-10, 10)] * len(variables)
            result = differential_evolution(objective_function, bounds)
        elif method == 'dual_annealing':
            bounds = [(-10, 10)] * len(variables)
            result = dual_annealing(objective_function, bounds)
        else:
            result = minimize(objective_function, initial_guess, method=method)

        # Extract the optimized coefficients
        optimized_coeffs = result.x
        optimized_coeffs_str = ', '.join(f'{coeff:.8f}' for coeff in optimized_coeffs)

        # Calculate final statistics with those optimized coeffs
        (mad, rmsd, mse, 
         min_error, max_error, 
         mape, mapd, 
         r_squared, mbe, 
         predicted, regression_slope, regression_intercept
        ) = calculate_mad(optimized_coeffs, df, variables, target)

        # -----------------------------
        # Cross-validation
        # -----------------------------
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_errors = []
        for train_index, test_index in kf.split(df):
            df_train = df.iloc[train_index]
            df_test = df.iloc[test_index]

            # Weighted sums
            weighted_sum_train = np.dot(df_train[variables], optimized_coeffs).reshape(-1, 1)
            weighted_sum_test  = np.dot(df_test[variables], optimized_coeffs).reshape(-1, 1)

            # OLS on the training data
            model_cv = LinearRegression()
            model_cv.fit(weighted_sum_train, df_train[target])

            # Predict test
            predicted_cv = model_cv.predict(weighted_sum_test)
            errors_cv = predicted_cv - df_test[target]

            mbe_cv = np.mean(errors_cv)
            mad_cv = np.mean(np.abs(errors_cv))
            mse_cv = np.mean(errors_cv)
            rmsd_cv = np.sqrt(np.mean(errors_cv**2))

            with np.errstate(divide='ignore', invalid='ignore'):
                percentage_errors_cv = np.abs(errors_cv / df_test[target])
                percentage_errors_cv = percentage_errors_cv[np.isfinite(percentage_errors_cv)]
                mape_cv = np.mean(percentage_errors_cv)*100 if len(percentage_errors_cv) > 0 else np.nan

            r_squared_cv = r2_score(df_test[target], predicted_cv)
            cv_errors.append((mad_cv, rmsd_cv, mse_cv, mape_cv, r_squared_cv, mbe_cv))

        # Average cross-validation errors
        cv_mad = np.mean([e[0] for e in cv_errors])
        cv_rmsd = np.mean([e[1] for e in cv_errors])
        cv_mse = np.mean([e[2] for e in cv_errors])
        cv_mape = np.mean([e[3] for e in cv_errors if not np.isnan(e[3])])
        cv_r_squared = np.mean([e[4] for e in cv_errors])
        cv_mbe = np.mean([e[5] for e in cv_errors])

        # -------------
        # Back-calculate final slopes/intercept in *original* domain
        # (the "Alternate Simplification" approach from your code)
        # -------------
        if not scale:
            # No scaling was used
            final_slopes = regression_slope * optimized_coeffs
            final_intercept = regression_intercept
        else:
            mu = scaler.mean_
            sigma = scaler.scale_
            # final_slopes_j = (regression_slope * optimized_coeffs_j) / sigma_j
            final_slopes = (regression_slope * optimized_coeffs) / sigma
            # final_intercept = regression_intercept - sum((regression_slope * optimized_coeffs_j * mu_j) / sigma_j)
            final_intercept = regression_intercept - np.sum(
                (regression_slope * optimized_coeffs * mu) / sigma
            )

        final_slopes_str = ', '.join(f'{fs:.8f}' for fs in final_slopes)

    # Prepare the result dictionary
    result_dict = {
        'mad': mad,
        'rmsd': rmsd,
        'mse': mse,
        'min_error': min_error,
        'max_error': max_error,
        'mape': mape,
        'mapd': mapd,
        'r_squared': r_squared,
        'mbe': mbe,
        'optimized_coeffs': optimized_coeffs,
        'optimized_coeffs_str': optimized_coeffs_str,
        'cv_mad': cv_mad,
        'cv_rmsd': cv_rmsd,
        'cv_mse': cv_mse,
        'cv_mape': cv_mape,
        'cv_r_squared': cv_r_squared,
        'cv_mbe': cv_mbe,
        'regression_slope': regression_slope,
        'regression_intercept': regression_intercept,
        'predicted': predicted,
        'final_intercept': final_intercept,
        'final_slopes': final_slopes,
        'final_slopes_str': final_slopes_str
    }
    '''
    # -------------
    # Plotting
    # -------------
    variables_str = '_'.join(variables)
    variables_hash = hashlib.md5(variables_str.encode()).hexdigest()[:8]
    method_label = 'direct' if direct_comparison else 'modeling'
    plot_prefix = f'plot_{target}_{variables_hash}_{method_label}'

    # Plot Predicted vs Actual
    plt.figure()
    plt.scatter(df[target], predicted, alpha=0.7)
    plt.xlabel('Actual')
    plt.ylabel('Predicted' if not direct_comparison else 'Variable Value')
    plt.title(f'Predicted vs Actual for {target}\nVariables: {variables}')
    plt.plot([df[target].min(), df[target].max()],
             [df[target].min(), df[target].max()],
             'r--')
    plt.tight_layout()
    plt.savefig(f'{plot_prefix}_pred_vs_actual.png')
    plt.close()

    # Plot Residuals
    plt.figure()
    plt.scatter(predicted, predicted - df[target], alpha=0.7)
    plt.xlabel('Predicted' if not direct_comparison else 'Variable Value')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted for {target}\nVariables: {variables}')
    plt.axhline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{plot_prefix}_residuals.png')
    plt.close()
    '''
    # Print results
    if direct_comparison:
        print(
            f"Direct Comparison - {target}: {variables}; "
            f"MAD: {mad:.8f}; RMSD: {rmsd:.8f}; MSE: {mse:.8f}; "
            f"Min Error: {min_error:.8f}; Max Error: {max_error:.8f}; "
            f"MAPE: {mape:.2f}%; MAPD: {mapd:.2f}%; R^2: {r_squared:.8f}; "
            f"MBE: {mbe:.8f}"
        )
    else:
        print(
            f"Modeling Approach - {target}: {variables}; "
            f"MAD: {mad:.8f}; RMSD: {rmsd:.8f}; MSE: {mse:.8f}; "
            f"Min Error: {min_error:.8f}; Max Error: {max_error:.8f}; "
            f"MAPE: {mape:.2f}%; MAPD: {mapd:.2f}%; R^2: {r_squared:.8f}; "
            f"MBE: {mbe:.8f}; CV MAD: {cv_mad:.8f}; CV RMSD: {cv_rmsd:.8f}; "
            f"CV R^2: {cv_r_squared:.8f}; Optimized Coefficients: {optimized_coeffs_str}; "
            f"Regression Slope (weighted sum): {regression_slope:.8f}; "
            f"Regression Intercept (weighted sum): {regression_intercept:.8f}; "
            f"Final Intercept (original scale): {final_intercept:.8f}; "
            f"Final Slopes (original scale): {final_slopes_str}"
        )

    return result_dict


###############################################################################
# process_data and __main__ remain the same; here's a quick skeleton for context
###############################################################################
def process_data(args):
    df, description = args
    df = df.copy()  # Ensure we're working with a copy
    plt.ioff()

    # Generate correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'Correlation Heatmap - {description}')
    plt.tight_layout()
    plt.savefig(f'corr_heatmap_{description.replace(" ", "_").replace(":", "")}.png')
    plt.close()

    # Write outputs to a file
    output_filename = f"output8R3_{description.replace(' ', '_').replace(':', '')}.txt"
    with open(output_filename, 'w') as f:
        f.write(f"\nProcessing {description}\n")
        f.write(f"Total number of entries: {df.shape[0]}\n")

#        possible_targets = [
#            'D4_rel_energy', 'homo_energy', 'lumo_energy',
#            'band_gap_eV', 'total_dipole', 'rel_energy_kcal_mol'
#        ]
        possible_targets = [
            'D4_rel_energy',
            'band_gap_eV',
            'total_dipole'
        ]
        for target in possible_targets:
            if target in df.columns:
                f.write(f"\nTarget: {target}\n")

                variables_list = [
                    # your variable sets
                    ['rel_energy_kcal_mol'],
                    ['xtb_rel_energy_kcal_mol'],
                    ['dihedral'],
                    ['sum_abs_120_minus_angle'],
                    ['rmsd_bond_angle'],
                    ['mean_bla'],
                    ['total_hydrogen_distance'],
                    ['mean_CC_distance'],
                    ['rmsd_bond_lengths'],
                    ['max_z_displacement'],
                    ['mean_baa'],
                    ['mean_z'],
                    ['rmsd_z'],
                    ['mad_z'],
                    ['mean_pyramidalization'],
                    ['rmsd_pyramidalization'],
                    ['total_dpo'],
                    ['max_mulliken_charge'],
                    ['min_mulliken_charge'],
                    ['area'],
                    ['max_cc_distance'],
                    ['asymmetry'],
                    ['longest_linear_path'],
                    ['wiener_val'],
                    ['harary_val'],
                    ['hyper_wiener_val'],
                    ['ecc_val'],
                    ['avg_homa_val'],
                    ['avg_homa2_val'],
                    ['avg_homa3_val'],
                    ['avg_homa4_val'],
                    ['avg_homa5_val'],
                    ['avg_homa6_val'],
                    ['avg_homa7_val'],
                    ['avg_homa8_val'],
                    ['avg_homa9_val'],
                    ['avg_homa10_val'],
                    ['avg_homa11_val'],
                    ['avg_homa12_val'],
                    ['avg_homa13_val'],
                    ['avg_homa14_val'],
                    ['avg_homa15_val'],
                    ['avg_homa16_val'],
                    ['avg_homa17_val'],
                    ['avg_homa18_val'],
                    ['avg_homa19_val'],
                    ['avg_homa20_val'],
                    ['avg_homa21_val'],
                    ['avg_homa22_val'],
                    ['avg_homa23_val'],
                    ['avg_homa24_val'],
                    ['avg_homa25_val'],
                    ['avg_homa26_val'],
                    ['radius_gyr_val'],
                    ['surf_val'],
                    ['I_princ_1_val'],
                    ['I_princ_2_val'],
                    ['I_princ_3_val'],
                    ['w3d_val'],
                    ['bay_val'],
                    ['dihedral', 'max_z_displacement'],
                    ['dihedral', 'mean_z'],
                    ['dihedral', 'rmsd_z'],
                    ['dihedral', 'mad_z'],
                    ['dihedral', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa_val'],
                    ['dihedral', 'avg_homa2_val'],
                    ['dihedral', 'avg_homa3_val'],
                    ['dihedral', 'avg_homa4_val'],
                    ['dihedral', 'avg_homa5_val'],
                    ['dihedral', 'avg_homa6_val'],
                    ['dihedral', 'avg_homa7_val'],
                    ['dihedral', 'avg_homa8_val'],
                    ['dihedral', 'avg_homa9_val'],
                    ['dihedral', 'avg_homa10_val'],
                    ['dihedral', 'avg_homa11_val'],
                    ['dihedral', 'avg_homa12_val'],
                    ['dihedral', 'avg_homa13_val'],
                    ['dihedral', 'avg_homa14_val'],
                    ['dihedral', 'avg_homa15_val'],
                    ['dihedral', 'avg_homa16_val'],
                    ['dihedral', 'avg_homa17_val'],
                    ['dihedral', 'avg_homa18_val'],
                    ['dihedral', 'avg_homa19_val'],
                    ['dihedral', 'avg_homa20_val'],
                    ['dihedral', 'avg_homa21_val'],
                    ['dihedral', 'avg_homa22_val'],
                    ['dihedral', 'avg_homa23_val'],
                    ['dihedral', 'avg_homa24_val'],
                    ['dihedral', 'avg_homa25_val'],
                    ['dihedral', 'avg_homa26_val'],
                    ['dihedral', 'rmsd_bond_angle'],
                    ['avg_homa_val','rmsd_bond_angle'],
                    ['avg_homa2_val','rmsd_bond_angle'],
                    ['avg_homa3_val','rmsd_bond_angle'],
                    ['avg_homa4_val','rmsd_bond_angle'],
                    ['avg_homa5_val','rmsd_bond_angle'],
                    ['avg_homa6_val','rmsd_bond_angle'],
                    ['avg_homa7_val','rmsd_bond_angle'],
                    ['avg_homa8_val','rmsd_bond_angle'],
                    ['avg_homa9_val','rmsd_bond_angle'],
                    ['avg_homa10_val','rmsd_bond_angle'],
                    ['avg_homa11_val','rmsd_bond_angle'],
                    ['avg_homa12_val','rmsd_bond_angle'],
                    ['avg_homa13_val','rmsd_bond_angle'],
                    ['avg_homa14_val','rmsd_bond_angle'],
                    ['avg_homa15_val','rmsd_bond_angle'],
                    ['avg_homa16_val','rmsd_bond_angle'],
                    ['avg_homa17_val','rmsd_bond_angle'],
                    ['avg_homa18_val','rmsd_bond_angle'],
                    ['avg_homa19_val','rmsd_bond_angle'],
                    ['avg_homa20_val','rmsd_bond_angle'],
                    ['avg_homa21_val','rmsd_bond_angle'],
                    ['avg_homa22_val','rmsd_bond_angle'],
                    ['avg_homa23_val','rmsd_bond_angle'],
                    ['avg_homa24_val','rmsd_bond_angle'],
                    ['avg_homa25_val','rmsd_bond_angle'],
                    ['avg_homa26_val','rmsd_bond_angle'],
                    ['avg_homa_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa2_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa3_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa4_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa5_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa6_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa7_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa8_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa9_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa10_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa11_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa12_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa13_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa14_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa15_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa16_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa17_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa18_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa19_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa20_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa21_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa22_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa23_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa24_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa25_val', 'xtb_rel_energy_kcal_mol'],
                    ['avg_homa26_val', 'xtb_rel_energy_kcal_mol'],
                    ['rmsd_bond_angle', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa2_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa3_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa4_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa5_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa6_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa7_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa8_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa9_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa10_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa11_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa12_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa13_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa14_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa15_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa16_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa17_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa18_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa19_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa20_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa21_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa22_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa23_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa24_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa25_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa26_val', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa2_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa3_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa4_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa5_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa6_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa7_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa8_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa9_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa10_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa11_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa12_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa13_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa14_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa15_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa16_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa17_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa18_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa19_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa20_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa21_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa22_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa23_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa24_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa25_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa26_val', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'avg_homa_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa2_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa3_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa4_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa5_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa6_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa7_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa8_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa9_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa10_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa11_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa12_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa13_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa14_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa15_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa16_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa17_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa18_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa19_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa20_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa21_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa22_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa23_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa24_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa25_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa26_val', 'rmsd_bond_angle'],
                    ['dihedral', 'avg_homa_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa2_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa3_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa4_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa5_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa6_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa7_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa8_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa9_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa10_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa11_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa12_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa13_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa14_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa15_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa16_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa17_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa18_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa19_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa20_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa21_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa22_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],\
                    ['dihedral', 'avg_homa23_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa24_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa25_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa26_val', 'rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'avg_homa_val','mean_baa'],
                    ['dihedral', 'avg_homa2_val','mean_baa'],
                    ['dihedral', 'avg_homa3_val','mean_baa'],
                    ['dihedral', 'avg_homa4_val','mean_baa'],
                    ['dihedral', 'avg_homa5_val','mean_baa'],
                    ['dihedral', 'avg_homa6_val','mean_baa'],
                    ['dihedral', 'avg_homa7_val','mean_baa'],
                    ['dihedral', 'avg_homa8_val','mean_baa'],
                    ['dihedral', 'avg_homa9_val','mean_baa'],
                    ['dihedral', 'avg_homa10_val','mean_baa'],
                    ['dihedral', 'avg_homa11_val','mean_baa'],
                    ['dihedral', 'avg_homa12_val','mean_baa'],
                    ['dihedral', 'avg_homa13_val','mean_baa'],
                    ['dihedral', 'avg_homa14_val','mean_baa'],
                    ['dihedral', 'avg_homa15_val','mean_baa'],
                    ['dihedral', 'avg_homa16_val','mean_baa'],
                    ['dihedral', 'avg_homa17_val','mean_baa'],
                    ['dihedral', 'avg_homa18_val','mean_baa'],
                    ['dihedral', 'avg_homa19_val','mean_baa'],
                    ['dihedral', 'avg_homa20_val','mean_baa'],
                    ['dihedral', 'avg_homa21_val','mean_baa'],
                    ['dihedral', 'avg_homa22_val','mean_baa'],
                    ['dihedral', 'avg_homa23_val','mean_baa'],
                    ['dihedral', 'avg_homa24_val','mean_baa'],
                    ['dihedral', 'avg_homa25_val','mean_baa'],
                    ['dihedral', 'avg_homa26_val','mean_baa'],
                    ['dihedral', 'wiener_val'],
                    ['dihedral', 'harary_val'],
                    ['dihedral', 'hyper_wiener_val'],
                    ['dihedral', 'ecc_val'],
                    ['dihedral', 'radius_gyr_val'],
                    ['dihedral', 'surf_val'],
                    ['dihedral', 'I_princ_1_val'],
                    ['dihedral', 'I_princ_2_val'],
                    ['dihedral', 'I_princ_3_val'],
                    ['dihedral', 'w3d_val'],
                    ['dihedral', 'bay_val'],
                    ['max_z_displacement', 'xtb_rel_energy_kcal_mol'],
                    ['mean_z', 'xtb_rel_energy_kcal_mol'],
                    ['rmsd_z', 'xtb_rel_energy_kcal_mol'],
                    ['mad_z', 'xtb_rel_energy_kcal_mol'],
                    ['mean_pyramidalization', 'xtb_rel_energy_kcal_mol'],
                    ['rmsd_pyramidalization', 'xtb_rel_energy_kcal_mol'],
                    ['sum_abs_120_minus_angle', 'xtb_rel_energy_kcal_mol'],
                    ['mean_bla', 'xtb_rel_energy_kcal_mol'],
                    ['mean_CC_distance', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'sum_abs_120_minus_angle'],
                    ['dihedral', 'mean_bla'],
                    ['dihedral', 'mean_CC_distance'],
                    ['sum_abs_120_minus_angle', 'mean_bla'],
                    ['dihedral', 'sum_abs_120_minus_angle', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'mean_bla', 'xtb_rel_energy_kcal_mol'],
                    ['sum_abs_120_minus_angle', 'mean_bla', 'xtb_rel_energy_kcal_mol'],
                    ['dihedral', 'sum_abs_120_minus_angle', 'mean_bla'],
                    ['dihedral', 'sum_abs_120_minus_angle', 'mean_bla', 'xtb_rel_energy_kcal_mol'],
                    # etc...
                ]

                print("shortened version")
                variables_list = []
                variables_list = [
                        ['avg_homa_val'],
                        ['dihedral','avg_homa_val'],
                        ['avg_homa_val','rmsd_bond_angle'],
                        ['avg_homa_val','xtb_rel_energy_kcal_mol'],
                        ['dihedral','avg_homa_val','sum_abs_120_minus_angle'],
                        ['dihedral','avg_homa_val','rmsd_bond_angle'],
                        ['dihedral','avg_homa_val','rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                        ['avg_homa_fused_val'],
                        ['dihedral','avg_homa_fused_val'],
                        ['avg_homa_fused_val','rmsd_bond_angle'],
                        ['avg_homa_fused_val','xtb_rel_energy_kcal_mol'],
                        ['dihedral','avg_homa_fused_val','sum_abs_120_minus_angle'],
                        ['dihedral','avg_homa_fused_val','rmsd_bond_angle'],
                        ['dihedral','avg_homa_fused_val','rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                        ['avg_homa_edge_val'],
                        ['dihedral','avg_homa_edge_val'],
                        ['avg_homa_edge_val','rmsd_bond_angle'],
                        ['avg_homa_edge_val','xtb_rel_energy_kcal_mol'],
                        ['dihedral','avg_homa_edge_val','sum_abs_120_minus_angle'],
                        ['dihedral','avg_homa_edge_val','rmsd_bond_angle'],
                        ['dihedral','avg_homa_edge_val','rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                        ['ring_homas_val'],
                        ['dihedral','ring_homas_val'],
                        ['ring_homas_val','rmsd_bond_angle'],
                        ['ring_homas_val','xtb_rel_energy_kcal_mol'],
                        ['dihedral','ring_homas_val','sum_abs_120_minus_angle'],
                        ['dihedral','ring_homas_val','rmsd_bond_angle'],
                        ['dihedral','ring_homas_val','rmsd_bond_angle','xtb_rel_energy_kcal_mol'],
                        ]

                variables_list = [
                        # avg_ring_homa
                        ['avg_ring_homa'],
                        ['dihedral',       'avg_ring_homa'],
                        ['avg_ring_homa',  'rmsd_bond_angle'],
                        ['avg_ring_homa',  'xtb_rel_energy_kcal_mol'],
                        ['dihedral',       'avg_ring_homa', 'sum_abs_120_minus_angle'],
                        ['dihedral',       'avg_ring_homa', 'rmsd_bond_angle'],
                        ['dihedral',       'avg_ring_homa','xtb_rel_energy_kcal_mol'],
                        ['dihedral',       'avg_ring_homa', 'rmsd_bond_angle', 'xtb_rel_energy_kcal_mol'],
                        
                        # global_homa
                        ['global_homa'],
                        ['dihedral',      'global_homa'],
                        ['global_homa',   'rmsd_bond_angle'],
                        ['global_homa',   'xtb_rel_energy_kcal_mol'],
                        ['dihedral',      'global_homa', 'sum_abs_120_minus_angle'],
                        ['dihedral',      'global_homa', 'rmsd_bond_angle'],
                        ['dihedral',       'global_homa','xtb_rel_energy_kcal_mol'],
                        ['dihedral',      'global_homa', 'rmsd_bond_angle', 'xtb_rel_energy_kcal_mol'],
                        
                        # weighted_homa
                        ['weighted_homa'],
                        ['dihedral',         'weighted_homa'],
                        ['weighted_homa',    'rmsd_bond_angle'],
                        ['weighted_homa',    'xtb_rel_energy_kcal_mol'],
                        ['dihedral',         'weighted_homa', 'sum_abs_120_minus_angle'],
                        ['dihedral',         'weighted_homa', 'rmsd_bond_angle'],
                        ['dihedral',       'weighted_homa','xtb_rel_energy_kcal_mol'],
                        ['dihedral',         'weighted_homa', 'rmsd_bond_angle', 'xtb_rel_energy_kcal_mol'],
                        
                        # fused_homa
                        ['fused_homa'],
                        ['dihedral',       'fused_homa'],
                        ['fused_homa',     'rmsd_bond_angle'],
                        ['fused_homa',     'xtb_rel_energy_kcal_mol'],
                        ['dihedral',       'fused_homa', 'sum_abs_120_minus_angle'],
                        ['dihedral',       'fused_homa', 'rmsd_bond_angle'],
                        ['dihedral',       'fused_homa','xtb_rel_energy_kcal_mol'],
                        ['dihedral',       'fused_homa', 'rmsd_bond_angle', 'xtb_rel_energy_kcal_mol'],
                        
                        # edge_homa
                        ['edge_homa'],
                        ['dihedral',      'edge_homa'],
                        ['edge_homa',     'rmsd_bond_angle'],
                        ['edge_homa',     'xtb_rel_energy_kcal_mol'],
                        ['dihedral',      'edge_homa', 'sum_abs_120_minus_angle'],
                        ['dihedral',      'edge_homa', 'rmsd_bond_angle'],
                        ['dihedral',       'edge_homa','xtb_rel_energy_kcal_mol'],
                        ['dihedral',      'edge_homa', 'rmsd_bond_angle', 'xtb_rel_energy_kcal_mol'],
                        ]



                for variables in variables_list:
                    if all(var in df.columns for var in variables):
                        try:
                            # For single var direct comparisons
                            perform_both = (
                                len(variables) == 1
                                and variables[0] in ['xtb_rel_energy_kcal_mol',
                                                     'rel_energy_kcal_mol']
                            )
                            if perform_both:
                                result_direct = minimize_mad(
                                    df, variables, target,
                                    direct_comparison=True
                                )
                                f.write(
                                    f"Direct Comparison - {target}: {variables}; "
                                    f"MAD: {result_direct['mad']:.8f}; "
                                    f"RMSD: {result_direct['rmsd']:.8f}; "
                                    f"MSE: {result_direct['mse']:.8f}; "
                                    f"MaxError: {result_direct['max_error']:.8f}; "
                                    f"R^2: {result_direct['r_squared']:.8f}\n"
                                )
                            
                            # Always do modeling approach
                            result_modeling = minimize_mad(
                                df, variables, target,
                                direct_comparison=False
                            )
                            f.write(
                                f"Modeling Approach - {target}: {variables}; "
                                f"MAD: {result_modeling['mad']:.8f}; "
                                f"RMSD: {result_modeling['rmsd']:.8f}; "
                                f"MSE: {result_modeling['mse']:.8f}; "
                                f"MaxError: {result_modeling['max_error']:.8f}; "
                                f"R^2: {result_modeling['r_squared']:.8f}; "
                                f"Final Intercept (orig)={result_modeling['final_intercept']:.8f}; "
                                f"Final Slopes (orig)={result_modeling['final_slopes_str']}\n"
                            )
                        except Exception as e:
                            f.write(f"Error with variables {variables}: {e}\n")
                    else:
                        missing_vars = [v for v in variables if v not in df.columns]
                        f.write(f"Skipping {variables} due to missing columns: {missing_vars}\n")


import os

def stem(fn):
    return os.path.splitext(fn)[0]

if __name__ == "__main__":
    # Example skeleton of reading files, etc...
    file_list = glob.glob('analysis_results.*.csv')
    dfs = []
    for file in file_list:
        df_temp = pd.read_csv(file)
        # ——— load the corresponding HOMA results and merge in ———
        PAH = file.split('.')[1]                      # e.g. "C36H20"
        homa_csv = f"{PAH}_homas.csv"                 # must exist: e.g. "C36H20_homas.csv"
        homa_df  = pd.read_csv(homa_csv)
        df_temp['stem'] = df_temp['file'].map(stem)
        homa_df['stem'] = homa_df['file'].map(stem)
        # merge on the 'file' column; we assume every xyz name appears in both
        df_temp = df_temp.merge(homa_df[['stem','avg_ring_homa','global_homa','weighted_homa','fused_homa','edge_homa']], on='stem', how='left').drop(columns='stem')
        df_temp['xtb_rel_energy_kcal_mol'] = (
            df_temp['xtb_raw_energy'] - df_temp['xtb_raw_energy'].min()
        ) * 627.509474
        df_temp['rel_energy_kcal_mol'] = (
            df_temp['energy_kcal_mol'] - df_temp['energy_kcal_mol'].min()
        )
        PAH_name = file.split('.')[1]
        df_temp['PAH'] = PAH_name
        df_temp.to_csv(file, index=False)
        dfs.append(df_temp)

    # Combine data in a new file (as you do)
    base_file = 'analysis_results.C36H20.csv'
    with open('analysis_results.COMPAS3_ALL.csv', 'w') as outfile:
        with open(base_file, 'r') as infile:
            outfile.write(infile.readline())
        for file in [
            'analysis_results.C36H20.csv',
            'analysis_results.C38H20.csv',
            'analysis_results.C40H20.csv',
            'analysis_results.C40H22.csv',
            'analysis_results.C42H22.csv',
            'analysis_results.C44H24.csv',
        ]:
            with open(file, 'r') as infile:
                next(infile)
                for line in infile:
                    outfile.write(line)

    all_data = pd.read_csv('analysis_results.COMPAS3_ALL.csv')
    all_data['dihedral'] = all_data['sum_less_90'] + all_data['sum_greater_90']

    all_data_filtered = all_data  # or exclude some rows, if desired

    # Build list of data subsets for parallel processing
    datasets_to_process = []
#    for PAH in ['COMPAS3_ALL']:
    for PAH in ['C36H20', 'C38H20', 'C40H20', 'C40H22', 'C42H22', 'C44H24','COMPAS3_ALL']:
        if PAH == 'COMPAS3_ALL':
            df_pah = all_data.copy()
        else:
            df_pah = all_data_filtered[all_data_filtered['PAH'] == PAH].copy()
#        df_pah = all_data.copy()
        df_pah['dihedral'] = df_pah['sum_less_90'] + df_pah['sum_greater_90']
        #df_pah['xtb_rel_energy_kcal_mol'] = (df_pah['xtb_raw_energy'] - df_pah['xtb_raw_energy'].min())*627.509474
        #df_pah['rel_energy_kcal_mol'] = (df_pah['energy_kcal_mol'] - df_pah['energy_kcal_mol'].min())
#Temp-while doing plainarity        datasets_to_process.append((df_pah, f"PAH_{PAH}"))

        df_pah['dihedral'] = df_pah['sum_less_90'] + df_pah['sum_greater_90']
        datasets_to_process.append((df_pah, f"PAH_{PAH}"))
        subsets = {
            'max_z_displacement_less_0.2': df_pah[df_pah['max_z_displacement'] < 0.2].copy(),
            'max_z_displacement_less_1': df_pah[df_pah['max_z_displacement'] < 1].copy(),
            'max_z_displacement_greater_1': df_pah[df_pah['max_z_displacement'] > 1].copy(),
            'max_z_displacement_greater_0.2': df_pah[df_pah['max_z_displacement'] > 0.2].copy(),
        }
        for subset_name, subset_df in subsets.items():
            datasets_to_process.append((subset_df, f"Subset_{PAH}_{subset_name}"))


    print(df_pah[['rel_energy_kcal_mol', 'xtb_rel_energy_kcal_mol',
                  'dihedral', 'sum_abs_120_minus_angle']].head())


    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_data, datasets_to_process)

