
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from dataclasses import dataclass

def visualize_warm_start(
        heur_sol: dataclass,
        time_steps: range,
        model_params: dataclass,
        save_path: str = None
):
    """Visualize the battery state of charge, charging, and discharging patterns
    for each battery type from the warm start solution. Also visualizes grid import/export
    prices and power flow.

    Args:
        charge (dict): Dictionary with battery charging values for each timestep
        discharge (dict): Dictionary with battery discharging values for each timestep
        soc (dict): Dictionary with state of charge values for each timestep
        time_steps (range): Range of optimization timesteps
        model_params (ModelParameters): Model parameters containing battery configurations
        grid_import (dict, optional): Dictionary with grid import values for each timestep
        grid_export (dict, optional): Dictionary with grid export values for each timestep
        save_path (str, optional): Path to save the plots. If None, plots are displayed.

    Returns:
        None
    """

    # Create a figure for each battery type
    for batt_type in model_params.battery_set:
        # Extract data for this battery
        soc_values = [heur_sol.soc.get((batt_type, t), 0) for t in time_steps]
        charge_values = [heur_sol.charge.get((batt_type, t), 0) for t in time_steps]
        discharge_values = [heur_sol.discharge.get((batt_type, t), 0) for t in time_steps]
        net_power = [charge_values[i] - discharge_values[i] for i in range(len(time_steps))]

        # Convert time steps to hour labels
        hours = list(time_steps)

        # Create figure with 2 subplots stacked vertically
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

        # Plot 1: State of Charge
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(hours, soc_values, 'b-', marker='o', linewidth=2, label='State of Charge (%)')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('State of Charge (%)')
        ax1.set_title(f'Battery {batt_type} - State of Charge over Time')
        ax1.grid(True)
        ax1.set_ylim([
            max(0, min(soc_values) - 5),  # Min with 5% padding
            min(100, max(soc_values) + 5)  # Max with 5% padding
        ])

        # Add horizontal lines for min and max SoC limits
        min_soc = model_params.soc_min.get(batt_type, 0)
        max_soc = model_params.soc_max.get(batt_type, 100)
        ax1.axhline(y=min_soc, color='r', linestyle='--', alpha=0.7, label=f'Min SoC ({min_soc}%)')
        ax1.axhline(y=max_soc, color='g', linestyle='--', alpha=0.7, label=f'Max SoC ({max_soc}%)')
        ax1.legend(loc='best')

        # Plot 2: Charging and Discharging Power
        ax2 = fig.add_subplot(gs[1])

        # Create bar chart for charge and discharge
        bar_width = 0.35
        x = np.arange(len(hours))

        # Plot charging as positive values
        charging_bars = ax2.bar(x - bar_width / 2, charge_values, bar_width, label='Charging Power (W)',
                                color='green', alpha=0.7)

        # Plot discharging as negative values
        discharge_bars = ax2.bar(x + bar_width / 2, discharge_values, bar_width, label='Discharging Power (W)',
                                 color='red', alpha=0.7)

        # Plot net power as a line
        ax2_twin = ax2.twinx()
        net_line = ax2_twin.plot(x, net_power, 'b-', marker='*', linewidth=2, label='Net Power (W)')

        # Add labels and legend for both axes
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Power (Watts)')
        ax2_twin.set_ylabel('Net Power (Watts)', color='blue')

        # Combine legends from both y-axes
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

        ax2.set_title(f'Battery {batt_type} - Charging and Discharging Power')
        ax2.set_xticks(x)
        ax2.set_xticklabels(hours)
        ax2.grid(True)

        # Add max power limit line
        max_power = model_params.power_max.get(batt_type, 0)
        ax2.axhline(y=max_power, color='purple', linestyle='-.', alpha=0.7,
                    label=f'Max Power ({max_power}W)')

        # Add annotations for capacity
        capacity = model_params.capacity.get(batt_type, 0)
        plt.figtext(0.02, 0.02, f"Battery Capacity: {capacity} Wh", fontsize=10)

        plt.tight_layout()

        # Save or show the plot
        if save_path:
            plt.savefig(f"{save_path}/battery_{batt_type}_visualization.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()

    # Create an additional plot showing all batteries' SoC on the same graph for comparison
    if len(model_params.battery_set) > 1:
        plt.figure(figsize=(12, 6))
        for batt_type in model_params.battery_set:
            soc_values = [heur_sol.soc.get((batt_type, t), 0) for t in time_steps]
            plt.plot(hours, soc_values, marker='o', linewidth=2, label=f'{batt_type} SoC (%)')

        plt.xlabel('Time (hours)')
        plt.ylabel('State of Charge (%)')
        plt.title('Comparison of All Batteries - State of Charge')
        plt.grid(True)
        plt.legend(loc='best')

        if save_path:
            plt.savefig(f"{save_path}/all_batteries_soc_comparison.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()

    # NEW CHART: Grid Import/Export with Prices
    if heur_sol.grid_import is not None and heur_sol.grid_export is not None:
        # Extract import/export data
        import_values = [heur_sol.grid_import.get(t, 0) for t in time_steps]
        export_values = [heur_sol.grid_export.get(t, 0) for t in time_steps]

        # Extract price data
        if isinstance(model_params.price_import, list):
            import_prices = [model_params.price_import[t] for t in time_steps]
        else:
            import_prices = [model_params.price_import for t in time_steps]

        if isinstance(model_params.price_export, list):
            export_prices = [model_params.price_export[t] for t in time_steps]
        else:
            export_prices = [model_params.price_export for t in time_steps]

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                       gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})

        # Plot 1: Grid Import/Export Power
        x = np.arange(len(hours))
        bar_width = 0.35

        import_bars = ax1.bar(x - bar_width / 2, import_values, bar_width, label='Grid Import (W)',
                              color='orange', alpha=0.7)
        export_bars = ax1.bar(x + bar_width / 2, export_values, bar_width, label='Grid Export (W)',
                              color='cyan', alpha=0.7)

        # Calculate net grid flow
        net_grid = [export_values[i] - import_values[i] for i in range(len(time_steps))]

        # Add net grid flow as a line on a twin axis
        ax1_twin = ax1.twinx()
        net_grid_line = ax1_twin.plot(x, net_grid, 'k-', marker='d', linewidth=2, label='Net Grid Flow (W)')

        # Add labels and legend
        ax1.set_ylabel('Power (Watts)')
        ax1_twin.set_ylabel('Net Grid Flow (Watts)', color='black')
        ax1.set_title('Grid Import and Export Power')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        ax1.set_xticks(x)
        ax1.set_xticklabels(hours)
        ax1.grid(True)

        # Plot 2: Import/Export Prices
        ax2.plot(x, import_prices, 'ro-', linewidth=2, label='Import Price (€/Wh)')
        ax2.plot(x, export_prices, 'go-', linewidth=2, label='Export Price (€/Wh)')

        # Add price difference as a filled area
        price_diff = [import_prices[i] - export_prices[i] for i in range(len(time_steps))]
        ax2.fill_between(x, 0, price_diff, alpha=0.2, color='purple', label='Price Difference')

        # Add labels and legend
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Price (€/Wh)')
        ax2.set_title('Grid Import and Export Prices')
        ax2.legend(loc='best')
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/grid_prices_and_flow.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()

        # Additional chart to show the correlation between grid actions and price differences
        plt.figure(figsize=(12, 6))

        # Calculate arbitrage metric (positive when charging during low prices and discharging during high prices)
        # For each battery type, calculate correlation metrics
        for batt_type in model_params.battery_set:
            charge_values = [heur_sol.charge.get((batt_type, t), 0) for t in time_steps]
            discharge_values = [heur_sol.discharge.get((batt_type, t), 0) for t in time_steps]
            net_battery_action = [discharge_values[i] - charge_values[i] for i in range(len(time_steps))]

            # Plot correlation between price difference and battery action
            plt.scatter(price_diff, net_battery_action, alpha=0.7, label=f'Battery {batt_type}')

        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

        # Annotate quadrants
        plt.text(max(price_diff) * 0.7, max([abs(v) for batt_type in model_params.battery_set for v in
                                             [heur_sol.discharge.get((batt_type, t), 0) - heur_sol.charge.get((batt_type, t), 0) for t
                                              in time_steps]]) * 0.7,
                 'Optimal: Discharge\nwhen import price > export price',
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

        plt.text(min(price_diff) * 0.7, -max([abs(v) for batt_type in model_params.battery_set for v in
                                              [heur_sol.discharge.get((batt_type, t), 0) - heur_sol.charge.get((batt_type, t), 0) for
                                               t in time_steps]]) * 0.7,
                 'Optimal: Charge\nwhen import price < export price',
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

        plt.xlabel('Price Difference (Import - Export) (€/Wh)')
        plt.ylabel('Net Battery Action (Discharge - Charge) (W)')
        plt.title('Price Arbitrage Analysis: Battery Actions vs. Price Differences')
        plt.grid(True)
        plt.legend(loc='best')

        if save_path:
            plt.savefig(f"{save_path}/price_arbitrage_analysis.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()