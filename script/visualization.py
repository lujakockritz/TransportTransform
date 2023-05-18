# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import networkx as nx
import matplotlib.cm as cm
#### VISUALIZATION ####
class viz_methods():
    def viz_modeCounts(mode_counts_list, runs, title, shading, text_lines):
        # Convert the mode_counts_list into a pandas dataframe
        df_mode_counts = pd.concat(mode_counts_list, axis=1, keys=range(runs))

        fig, ax = plt.subplots(figsize=(10, 6))

        # Set axis labels and title
        ax.set_xlabel('Time Step', fontsize=14)
        ax.set_ylabel('Number of agents', fontsize=14)
        ax.set_title(title, fontsize=16)

        # Define a dictionary of colors for each mode
        colors = {'d_car': '#1F77B4', # Dark blue
                'd_public': '#2CA02C', # Dark green
                'd_bike': '#FF7F0E', # Dark orange
                'd_carsh': '#9467BD'} # Dark purple
        
        mode_names = {
            'd_car': 'Car',
            'd_public': 'Public Transport',
            'd_bike': 'Bike',
            'd_carsh': 'Carsharing'}


        if runs == 1:
            for mode in df_mode_counts.columns.levels[2]:
                mode_data = df_mode_counts[mode]
                data_min = mode_data.min(axis=1)
                data_max = mode_data.max(axis=1)
                if shading:
                    ax.fill_between(data_min.index, data_min, data_max, color=colors[mode], alpha=0.2)
                
                # Plot median line for each mode
                data_median = mode_data.median(axis=1)
                ax.plot(mode_data.index, mode_data, color=colors[mode], label=f'{mode_names[mode]}')
        
        else:
            # Shade the region between the minimum and maximum values for each mode
            for mode in df_mode_counts.columns.levels[2]:
                mode_data = df_mode_counts.iloc[:, df_mode_counts.columns.get_level_values(2) == mode]
                data_min = mode_data.min(axis=1)
                data_max = mode_data.max(axis=1)
                if shading:
                    ax.fill_between(data_min.index, data_min, data_max, color=colors[mode], alpha=0.2)
                
                # Plot median line for each mode
                data_median = mode_data.median(axis=1)
                ax.plot(data_median.index, data_median, color=colors[mode], label=f'{mode_names[mode]}')

        # Add a legend
        legend_handles = []
        if shading:
            # Add subheading for mean lines
            legend_handles.append(mlines.Line2D([], [], color='none', linestyle='none', label='Mean across runs'))


        # Add legend handles for each mode
        for mode in ['d_car', 'd_bike', 'd_public','d_carsh']:
            legend_handles.append(
                mlines.Line2D([], [], color=colors[mode], linestyle='-', 
                              label=f'{mode_names[mode]}'))
            
        if shading:
            # Add an empty row in the legend
            legend_handles.append(mpatches.Patch(facecolor='none', edgecolor='none', label=''))

            # Add subheading for shaded areas
            legend_handles.append(mpatches.Patch(facecolor='none', edgecolor='none', label='Range across runs'))


            # Add legend handles for each mode area
            for mode in ['d_car', 'd_bike', 'd_public', 'd_carsh']:
                legend_handles.append(
                    mpatches.Patch(facecolor=colors[mode], alpha=0.2, label=f'{mode_names[mode]}')
                )

        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), title="Legend", 
                  alignment='left' , title_fontsize='large', prop={'size': 12})
        
        if text_lines is not None:
            text = '\n'.join(text_lines)
            bbox_props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)
            ax.text(1.02, 0.02, text, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=bbox_props)

        
        # Set x-axis ticks to integers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def visualize_graph_pref_1(model):
        # Create a list of node colors based on the agents' preference_1 attribute
        node_preferences = [agent.preference_1 for agent in model.schedule.agents]

        colors = {'d_car': '#1F77B4', # Dark blue
                'd_public': '#2CA02C', # Dark green
                'd_bike': '#FF7F0E', # Dark orange
                'd_carsh': '#9467BD'} # Dark purple

        # Map the attribute values to colors using a color map
        node_colors = [colors[val] for val in node_preferences]

        # Set the node size based on the number of nodes in the graph
        num_nodes = model.G.number_of_nodes()
        node_size = 2500.0 / num_nodes

        # Set the edge thickness based on the number of edges in the graph
        num_edges = model.G.number_of_edges()
        edge_thickness = 2.5 / (num_edges ** 0.5)

        # Draw the graph with node colors
        pos = nx.spring_layout(model.G)
        nx.draw_networkx_nodes(model.G, pos, node_color=node_colors, node_size=node_size)
        nx.draw_networkx_edges(model.G, pos, width=edge_thickness)

        mode_names = {
            'd_car': 'Car',
            'd_public': 'Public Transport',
            'd_bike': 'Bike',
            'd_carsh': 'Carsharing'}

        # Add a legend
        legend_handles = []
        for mode in ['d_car', 'd_bike', 'd_public', 'd_carsh']:
            legend_handles.append(
                mpatches.Patch(color=colors[mode], label=mode_names[mode])
            )

        plt.legend(handles=legend_handles, loc='lower right', title="Legend", 
                  alignment='left' , title_fontsize='small', prop={'size': 8})

        plt.title('Network visualization based on first mode preference')

        plt.axis('off')
        plt.show()    

    def visualize_big_graph_pref_1(model):
        # Create a list of node colors based on the agents' preference_1 attribute
        node_preferences = [agent.preference_1 for agent in model.schedule.agents]

        colors = {'d_car': '#1F77B4', # Dark blue
                'd_public': '#2CA02C', # Dark green
                'd_bike': '#FF7F0E', # Dark orange
                'd_carsh': '#9467BD'} # Dark purple

        # Map the attribute values to colors using a color map
        node_colors = [colors[val] for val in node_preferences]

        # Set the node size based on the number of nodes in the graph
        num_nodes = model.G.number_of_nodes()
        node_size = 2500.0 / num_nodes

        # Set the edge thickness based on the number of edges in the graph
        num_edges = model.G.number_of_edges()
        edge_thickness = 2.5 / (num_edges ** 0.5)

        # Draw the graph with node colors
        pos = nx.spring_layout(model.G)
        nx.draw_networkx_nodes(model.G, pos, node_color=node_colors, node_size=node_size)
        nx.draw_networkx_edges(model.G, pos, width=edge_thickness)

        plt.title('Network visualization based on first mode preference')

        plt.axis('off')
        plt.savefig('graph_without_legend.png', dpi=300, bbox_inches='tight')

    def edit_big_fig():
            # Clear the current plot
        plt.clf()

        # Load the saved figure and add the legend
        img = plt.imread('graph_without_legend.png')
        plt.imshow(img, aspect='equal')
        plt.axis('off')

        mode_names = {
            'd_car': 'Car',
            'd_public': 'Public Transport',
            'd_bike': 'Bike',
            'd_carsh': 'Carsharing'
        }

        colors = {'d_car': '#1F77B4', # Dark blue
                'd_public': '#2CA02C', # Dark green
                'd_bike': '#FF7F0E', # Dark orange
                'd_carsh': '#9467BD'} # Dark purple

        # Add a legend
        legend_handles = []
        for mode in ['d_car', 'd_bike', 'd_public', 'd_carsh']:
            legend_handles.append(
                mpatches.Patch(color=colors[mode], label=mode_names[mode])
            )

        plt.legend(handles=legend_handles, loc='lower right', title="Legend", 
                  alignment='left' , title_fontsize='small', prop={'size': 8})

        plt.show()        