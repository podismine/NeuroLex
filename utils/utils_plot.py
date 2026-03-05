import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_centiles(model, test_data, train_data, centiles=[3, 10, 25, 50, 75, 90, 97], 
                    func_name='SHASH', affected=None, variables_to_split=None, 
                    x_axis={'age': 'Age'}, additional_global_vars=None, 
                    plot_scatter=True, plot_all_data=False, subset_scatter=None, site_correction=False, 
                    correction_col_name='site', figsize=(8, 5), save_image=None, file_name=None, 
                    dpi=300, font_size=[12, 10, 10, 8],show_fig = False):
    import os
     
    def is_categorical(series):
        return series.dtype == 'object' or series.nunique() < 10

    x_col = list(x_axis.keys())[0]
    global_stats = {}

    # Get all variables that need fixed values (all x_vals except x-axis and variables_to_split)
    variables_to_fix = []
    split_vars = set(variables_to_split.keys()) if variables_to_split else set()

    for var in model.x_vals:
        if var != x_col and var not in split_vars:
            variables_to_fix.append(var)

    # Calculate global stats for each variable
    for var in variables_to_fix:
        if var in test_data.columns:
            # Check if user provided a specific value
            if additional_global_vars and var in additional_global_vars:
                global_stats[var] = additional_global_vars[var]
                print(f"{var}: user specified = {global_stats[var]}")
            else:
                # Auto-calculate mode for categorical, mean for continuous
                if is_categorical(test_data[var]):
                    global_stats[var] = test_data[var].mode().iloc[0]
                    print(f"{var}: mode = {global_stats[var]}")
                else:
                    global_stats[var] = test_data[var].mean()
                    print(f"{var}: mean = {global_stats[var]:.3f}")

    def get_grid_data(test_data, fixed_values=None):
        x_col = list(x_axis.keys())[0]
        x_min, x_max = test_data[x_col].min(), test_data[x_col].max()
        x_grid = np.linspace(x_min, x_max, 200)
        
        # Initialize with a row from test_data to keep original values
        grid_data = pd.DataFrame([test_data.iloc[0].copy() for _ in range(len(x_grid))]).reset_index(drop=True)
        
        # Set x-axis values
        grid_data[x_col] = x_grid
        
        # Update fixed values (from variables_to_split)
        if fixed_values:
            for col, val in fixed_values.items():
                grid_data[col] = val
        
        # Update ALL global stats (automatically calculated + user specified)
        for col, val in global_stats.items():
            grid_data[col] = val
        
        return grid_data

    def get_scatter_data(test_data, grid_data, combination=None):
        """
        Get the appropriate test data for scatter plotting based on plot_all_data and site_correction settings.
        """
        if plot_all_data:
            scatter_data = test_data.copy()
            
            if site_correction:
                # Get reference site from grid_data
                reference_site = grid_data[correction_col_name].iloc[0]
                # Apply site correction
                scatter_data = model.create_site_adjusted_data(scatter_data, reference_site, correction_col_name)
                # Use the adjusted y values
                scatter_data = scatter_data.copy()
                scatter_data[model.y_val] = scatter_data[f'{model.y_val}_adjusted']
        else:
            # Filter test data based on subset_scatter and variables_to_split only
            scatter_data = test_data.copy()
            
            # Apply subset_scatter filtering (only specified variables from global_stats)
            if subset_scatter:
                for var in subset_scatter:
                    if var in global_stats and var in scatter_data.columns:
                        scatter_data = scatter_data[scatter_data[var] == global_stats[var]]
            
            # Apply variables_to_split filtering for current combination
            if combination:
                for col, val in combination.items():
                    scatter_data = scatter_data[scatter_data[col] == val]
        
        return scatter_data

    def plot_single(grid_data, test_data, train_data, title_suffix, combination=None):
        ret = dict()
        x_col = list(x_axis.keys())[0]
        x_label = list(x_axis.values())[0]

        centile_values = model.centiles(grid_data, train_data, centiles, func_name)

        plt.figure(figsize=figsize)
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(centiles)))
        for i, (centile, color) in enumerate(zip(centiles, colors)):
            plt.plot(grid_data[x_col], centile_values[:, i], color=color, linestyle='-', label=f'{model._get_ordinal_suffix(centile)} Centile')

        ret['x'] = grid_data[x_col]
        ret['centile'] = centile_values
        ret['centile_num'] = centiles

        # Only plot scatter points if plot_scatter is True
        if plot_scatter:
            # Get appropriate scatter data
            scatter_data = get_scatter_data(test_data, grid_data, combination)
            
            plt.scatter(scatter_data[x_col], scatter_data[model.y_val], color='blue', s=20, facecolors='none', 
                    edgecolors='skyblue', label='Actual Values')
            ret['scatter_x'] = scatter_data[x_col]
            ret['scatter_y'] = scatter_data[model.y_val]

            if affected is not None:
                # Filter affected array to match scatter_data indices
                affected_indices = scatter_data.index
                affected_filtered = affected[affected_indices]
                
                unique_labels = pd.Series(affected_filtered).dropna().unique()
                color_map = plt.cm.get_cmap('tab20', len(unique_labels))
                
                for i, label in enumerate(unique_labels):
                    if pd.notna(label) and label != 'None':
                        label_mask = affected_filtered == label
                        color = color_map(i)
                        
                        if label_mask.sum() > 0:
                            plot_x = scatter_data.loc[label_mask, x_col]
                            plot_y = scatter_data.loc[label_mask, model.y_val]
                            plt.scatter(plot_x, plot_y, facecolors='none', edgecolors=color, marker='o', s=50, label=label)
                            ret[f'{label}_x'] = plot_x
                            ret[f'{label}_y'] = plot_y
        # Set custom font sizes
        plt.xlabel(x_label, fontsize=font_size[1])
        plt.ylabel(model.y_val, fontsize=font_size[2])
        plt.title(f'Centile Plot for {model.y_val} {title_suffix}', fontsize=font_size[0])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size[3])
        plt.grid(True)
        plt.tight_layout()
        if show_fig:
            plt.show()

        print("\nValues used for centile calculation in this plot:")
        
        # Print variables_to_split values for this combination
        if combination:
            for var, val in combination.items():
                print(f"{var}: {val}")

        for var in model.x_vals:
            if var != x_col and (not combination or var not in combination):
                if var in global_stats:
                    print(f"{var}: {global_stats[var]}")
        
        # Print scatter filtering info if plot_all_data is False
        if not plot_all_data and plot_scatter:
            print(f"\nScatter data filtered by:")
            if subset_scatter:
                print(f"  subset_scatter variables: {subset_scatter}")
                for var in subset_scatter:
                    if var in global_stats:
                        print(f"    {var}: {global_stats[var]}")
            if combination:
                print(f"  variables_to_split:")
                for var, val in combination.items():
                    print(f"    {var}: {val}")
            if not subset_scatter and not combination:
                print("  No filtering applied (showing all scatter data)")

        
        return ret
    

    def get_dict(grid_data, test_data, train_data, title_suffix, combination=None):
        ret = dict()
        x_col = list(x_axis.keys())[0]
        x_label = list(x_axis.values())[0]

        centile_values = model.centiles(grid_data, train_data, centiles, func_name)
        ret['x'] = grid_data[x_col]
        ret['centile'] = centile_values
        ret['centile_num'] = centiles
        if plot_scatter:
            # Get appropriate scatter data
            scatter_data = get_scatter_data(test_data, grid_data, combination)
            
            ret['scatter_x'] = scatter_data[x_col]
            ret['scatter_y'] = scatter_data[model.y_val]

            if affected is not None:
                affected_indices = scatter_data.index
                affected_filtered = affected[affected_indices]
                
                unique_labels = pd.Series(affected_filtered).dropna().unique()
                color_map = plt.cm.get_cmap('tab20', len(unique_labels))
                
                for i, label in enumerate(unique_labels):
                    if pd.notna(label) and label != 'None':
                        label_mask = affected_filtered == label
                        color = color_map(i)
                        if label_mask.sum() > 0:
                            plot_x = scatter_data.loc[label_mask, x_col]
                            plot_y = scatter_data.loc[label_mask, model.y_val]
                            ret[f'{label}_x'] = plot_x
                            ret[f'{label}_y'] = plot_y
        return ret
    
    if affected is not None:
        if len(test_data) != len(affected):
            raise ValueError("Length mismatch: 'affected' array length must match test_data length")

    if variables_to_split is None:
        grid_data = get_grid_data(test_data)
        if show_fig is True:
            plot_single(grid_data, test_data, train_data, "", None)
        plot_data = get_dict(grid_data, test_data, train_data, "", None)
        return plot_data
    else:
        ret = []
        combinations = [{}]
        for var, values in variables_to_split.items():
            combinations = [dict(comb, **{var: val}) for comb in combinations for val in values]

        for combination in combinations:
            filtered_data = test_data.copy()
            title_parts = []
            for var, val in combination.items():
                filtered_data = filtered_data[filtered_data[var] == val]
                title_parts.append(f"{var}={val}")

            if len(filtered_data) > 0:
                grid_data = get_grid_data(filtered_data, fixed_values=combination)
                if show_fig is True:
                    plot_single(grid_data, filtered_data, train_data, f"({', '.join(title_parts)})", combination)
                plot_data = get_dict(grid_data, filtered_data, train_data, f"({', '.join(title_parts)})", combination)
                ret.append(plot_data)
            else:
                print(f"No data for combination: {combination}")
        return ret