from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from dataclasses import dataclass, field

from akkudoktoreos.optimization.genetic import OptimizationParameters


def visualize_warm_start(
    heur_sol: dataclass,
    time_steps: range,
    model_params: dataclass,
    save_path: str = None,
):
    """Visualize battery state data and grid interactions from warm start solution.

    This function creates multiple visualizations to analyze battery behavior and grid
    interactions. It generates plots for each battery type showing state of charge trends,
    charging/discharging patterns, grid import/export with pricing data, and price
    arbitrage analysis.

    Parameters
    ----------
    heur_sol : dataclass
        A dataclass containing the warm start solution with the following attributes:
            - soc : dict
                Dictionary with battery state of charge values, keyed by (battery_type, timestep)
            - charge : dict
                Dictionary with battery charging values, keyed by (battery_type, timestep)
            - discharge : dict
                Dictionary with battery discharging values, keyed by (battery_type, timestep)
            - grid_import : dict
                Dictionary with grid import values, keyed by timestep
            - grid_export : dict
                Dictionary with grid export values, keyed by timestep
            - flow_direction : dict, optional
                Dictionary indicating flow direction (may be None)
    time_steps : range
        Range object defining the optimization time horizon
    model_params : dataclass
        ModelParameters object containing battery configurations with the following attributes:
            - battery_set : list
                List of available battery types in the system
            - soc_min : dict
                Minimum state of charge allowed for each battery type
            - soc_max : dict
                Maximum state of charge allowed for each battery type
            - power_max : dict
                Maximum charging/discharging power for each battery type
            - capacity : dict
                Energy storage capacity for each battery type
            - price_import : list or float
                Import prices for each timestep or constant value
            - price_export : list or float
                Export prices for each timestep or constant value
    save_path : str, optional
        Path to save the generated plots. If None, plots are displayed
        interactively, by default None

    Returns:
    -------
    None
        This function does not return any value but generates and displays/saves plots

    Notes:
    -----
    The function creates multiple types of visualizations:
    - Individual battery plots showing SoC, charging, and discharging patterns
    - Comparison plot for all batteries' SoC (if multiple batteries exist)
    - Grid import/export with pricing data
    - Price arbitrage analysis correlating battery actions with price differences
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
        ax1.plot(
            hours,
            soc_values,
            "b-",
            marker="o",
            linewidth=2,
            label="State of Charge (%)",
        )
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("State of Charge (%)")
        ax1.set_title(f"Battery {batt_type} - State of Charge over Time")
        ax1.grid(True)
        ax1.set_ylim(
            [
                max(0, min(soc_values) - 5),  # Min with 5% padding
                min(100, max(soc_values) + 5),  # Max with 5% padding
            ]
        )

        # Add horizontal lines for min and max SoC limits
        min_soc = model_params.soc_min.get(batt_type, 0)
        max_soc = model_params.soc_max.get(batt_type, 100)
        ax1.axhline(
            y=min_soc,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Min SoC ({min_soc}%)",
        )
        ax1.axhline(
            y=max_soc,
            color="g",
            linestyle="--",
            alpha=0.7,
            label=f"Max SoC ({max_soc}%)",
        )
        ax1.legend(loc="best")

        # Plot 2: Charging and Discharging Power
        ax2 = fig.add_subplot(gs[1])

        # Create bar chart for charge and discharge
        bar_width = 0.35
        x = np.arange(len(hours))

        # Plot charging as positive values
        charging_bars = ax2.bar(
            x - bar_width / 2,
            charge_values,
            bar_width,
            label="Charging Power (W)",
            color="green",
            alpha=0.7,
        )

        # Plot discharging as negative values
        discharge_bars = ax2.bar(
            x + bar_width / 2,
            discharge_values,
            bar_width,
            label="Discharging Power (W)",
            color="red",
            alpha=0.7,
        )

        # Plot net power as a line
        ax2_twin = ax2.twinx()
        net_line = ax2_twin.plot(x, net_power, "b-", marker="*", linewidth=2, label="Net Power (W)")

        # Add labels and legend for both axes
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Power (Watts)")
        ax2_twin.set_ylabel("Net Power (Watts)", color="blue")

        # Combine legends from both y-axes
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

        ax2.set_title(f"Battery {batt_type} - Charging and Discharging Power")
        ax2.set_xticks(x)
        ax2.set_xticklabels(hours)
        ax2.grid(True)

        # Add max power limit line
        max_power = model_params.power_max.get(batt_type, 0)
        ax2.axhline(
            y=max_power,
            color="purple",
            linestyle="-.",
            alpha=0.7,
            label=f"Max Power ({max_power}W)",
        )

        # Add annotations for capacity
        capacity = model_params.capacity.get(batt_type, 0)
        plt.figtext(0.02, 0.02, f"Battery Capacity: {capacity} Wh", fontsize=10)

        plt.tight_layout()

        # Save or show the plot
        if save_path:
            plt.savefig(
                f"{save_path}/battery_{batt_type}_visualization.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.show()

    # Create an additional plot showing all batteries' SoC on the same graph for comparison
    if len(model_params.battery_set) > 1:
        plt.figure(figsize=(12, 6))
        for batt_type in model_params.battery_set:
            soc_values = [heur_sol.soc.get((batt_type, t), 0) for t in time_steps]
            plt.plot(hours, soc_values, marker="o", linewidth=2, label=f"{batt_type} SoC (%)")

        plt.xlabel("Time (hours)")
        plt.ylabel("State of Charge (%)")
        plt.title("Comparison of All Batteries - State of Charge")
        plt.grid(True)
        plt.legend(loc="best")

        if save_path:
            plt.savefig(
                f"{save_path}/all_batteries_soc_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
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
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(12, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1], "hspace": 0.3},
        )

        # Plot 1: Grid Import/Export Power
        x = np.arange(len(hours))
        bar_width = 0.35

        import_bars = ax1.bar(
            x - bar_width / 2,
            import_values,
            bar_width,
            label="Grid Import (W)",
            color="orange",
            alpha=0.7,
        )
        export_bars = ax1.bar(
            x + bar_width / 2,
            export_values,
            bar_width,
            label="Grid Export (W)",
            color="cyan",
            alpha=0.7,
        )

        # Calculate net grid flow
        net_grid = [export_values[i] - import_values[i] for i in range(len(time_steps))]

        # Add net grid flow as a line on a twin axis
        ax1_twin = ax1.twinx()
        net_grid_line = ax1_twin.plot(
            x, net_grid, "k-", marker="d", linewidth=2, label="Net Grid Flow (W)"
        )

        # Add labels and legend
        ax1.set_ylabel("Power (Watts)")
        ax1_twin.set_ylabel("Net Grid Flow (Watts)", color="black")
        ax1.set_title("Grid Import and Export Power")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        ax1.set_xticks(x)
        ax1.set_xticklabels(hours)
        ax1.grid(True)

        # Plot 2: Import/Export Prices
        ax2.plot(x, import_prices, "ro-", linewidth=2, label="Import Price (€/Wh)")
        ax2.plot(x, export_prices, "go-", linewidth=2, label="Export Price (€/Wh)")

        # Add price difference as a filled area
        price_diff = [import_prices[i] - export_prices[i] for i in range(len(time_steps))]
        ax2.fill_between(x, 0, price_diff, alpha=0.2, color="purple", label="Price Difference")

        # Add labels and legend
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Price (€/Wh)")
        ax2.set_title("Grid Import and Export Prices")
        ax2.legend(loc="best")
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/grid_prices_and_flow.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

        # Additional chart to show the correlation between grid actions and price differences
        plt.figure(figsize=(12, 6))

        # Calculate arbitrage metric (positive when charging during low prices and discharging during high prices)
        # For each battery type, calculate correlation metrics
        for batt_type in model_params.battery_set:
            charge_values = [heur_sol.charge.get((batt_type, t), 0) for t in time_steps]
            discharge_values = [heur_sol.discharge.get((batt_type, t), 0) for t in time_steps]
            net_battery_action = [
                discharge_values[i] - charge_values[i] for i in range(len(time_steps))
            ]

            # Plot correlation between price difference and battery action
            plt.scatter(price_diff, net_battery_action, alpha=0.7, label=f"Battery {batt_type}")

        plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

        # Annotate quadrants
        plt.text(
            max(price_diff) * 0.7,
            max(
                [
                    abs(v)
                    for batt_type in model_params.battery_set
                    for v in [
                        heur_sol.discharge.get((batt_type, t), 0)
                        - heur_sol.charge.get((batt_type, t), 0)
                        for t in time_steps
                    ]
                ]
            )
            * 0.7,
            "Optimal: Discharge\nwhen import price > export price",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        plt.text(
            min(price_diff) * 0.7,
            -max(
                [
                    abs(v)
                    for batt_type in model_params.battery_set
                    for v in [
                        heur_sol.discharge.get((batt_type, t), 0)
                        - heur_sol.charge.get((batt_type, t), 0)
                        for t in time_steps
                    ]
                ]
            )
            * 0.7,
            "Optimal: Charge\nwhen import price < export price",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        plt.xlabel("Price Difference (Import - Export) (€/Wh)")
        plt.ylabel("Net Battery Action (Discharge - Charge) (W)")
        plt.title("Price Arbitrage Analysis: Battery Actions vs. Price Differences")
        plt.grid(True)
        plt.legend(loc="best")

        if save_path:
            plt.savefig(
                f"{save_path}/price_arbitrage_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.show()


@dataclass
class ModelParameters:
    """Model for grid battery storage systems with multiple battery types.

    This class manages parameters for modeling grid-connected battery storage systems
    with support for multiple battery types. It handles state of charge (SoC) constraints,
    power limits, efficiencies, and economic parameters for optimization.

    Attributes:
    ----------
    soc_min : Dict[str, float]
        Minimum state of charge allowed for each battery type (percentage 0-100)
    soc_max : Dict[str, float]
        Maximum state of charge allowed for each battery type (percentage 0-100)
    soc_init : Dict[str, float]
        Initial state of charge for each battery type (percentage 0-100)
    power_max : Dict[str, float]
        Maximum charging/discharging power for each battery type (watts)
    capacity : Dict[str, float]
        Energy storage capacity for each battery type (watt-hours)
    eff_charge : Dict[str, float]
        Charging efficiency for each battery type (decimal 0-1),
        representing energy stored / energy input
    eff_discharge : Dict[str, float]
        Discharging efficiency for each battery type (decimal 0-1),
        representing energy output / energy discharged
    battery_set : List[str]
        List of available battery types in the system
    total_load : List[float]
        Required total energy demand for each timestep
    pv_forecast : List[float]
        Forecasted PV production for each timestep (watt-hours)
    price_import : List[float]
        Price for buying electricity from grid for each timestep (€/Wh)
    price_export : List[float]
        Price for selling electricity to grid for each timestep (€/Wh)
    price_storage : float
        Value of stored energy at end of optimization horizon (€/Wh)
    no_discharge : List[float]
        Periods where battery discharge is not allowed
    """

    # Minimum state of charge allowed for each battery type (in percentage 0-100)
    soc_min: Dict[str, float] = field(default_factory=dict)

    # Maximum state of charge allowed for each battery type (in percentage 0-100)
    soc_max: Dict[str, float] = field(default_factory=dict)

    # Initial state of charge for each battery type (in percentage 0-100)
    soc_init: Dict[str, float] = field(default_factory=dict)

    # Maximum charging/discharging power for each battery type (in watts)
    power_max: Dict[str, float] = field(default_factory=dict)

    # Energy storage capacity for each battery type (in watt-hours)
    capacity: Dict[str, float] = field(default_factory=dict)

    # Charging efficiency for each battery type (as decimal 0-1)
    # Represents energy stored / energy input
    eff_charge: Dict[str, float] = field(default_factory=dict)

    # Discharging efficiency for each battery type (as decimal 0-1)
    # Represents energy output / energy discharged
    eff_discharge: Dict[str, float] = field(default_factory=dict)

    # List of available battery types in the system
    battery_set: List[str] = field(default_factory=list)

    total_load: List[float] = field(default_factory=list)
    pv_forecast: List[float] = field(default_factory=list)
    price_import: List[float] = field(default_factory=list)
    price_export: List[float] = field(default_factory=list)
    price_storage: float = field(default_factory=list)
    can_discharge: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def init_from_parameters(cls, parameters: OptimizationParameters, config):
        """Initialize GridModel from optimization parameters.

        Parameters
        ----------
        parameters : OptimizationParameters
            Object containing battery configurations and EMS parameters
            Expected to have attributes such as 'pv_akku', 'eauto', and 'ems'

        Returns:
        -------
        ModelParameters
            New instance populated with battery and grid parameters

        Notes:
        -----
        This method extracts relevant parameters from the input object and
        creates a properly formatted ModelParameters instance ready for
        optimization algorithms.
        """
        grid_model = cls()

        # time horizon under consideration
        time_steps = config.optimization_hours

        # Extract parameters from input
        grid_model.add_ems_parameters(parameters, time_steps)

        # Define supported battery types
        battery_types = ["pv_akku", "eauto"]  # PV battery storage and electric vehicle

        # Add each battery type if it exists in parameters
        for batt_type in battery_types:
            grid_model.add_battery(parameters, batt_type)

        return grid_model

    def add_ems_parameters(self, parameters: OptimizationParameters, time_steps: int):
        """Translate OptimizationParameters into model format.

        This function converts parameters from the OptimizationParameters format
        into the internal model format required for the optimization problem formulation.

        Parameters
        ----------
        parameters : OptimizationParameters
            Object containing EMS parameters with attributes:
                - ems.gesamtlast : Required total energy for each timestep
                - ems.pv_prognose_wh : Forecasted PV production
                - ems.strompreis_euro_pro_wh : Price for buying electricity from grid
                - ems.einspeiseverguetung_euro_pro_wh : Price for selling to grid
                - ems.preis_euro_pro_wh_akku : Value of stored energy at end of horizon

        Returns:
        -------
        None
            This method modifies the current instance in-place
        """
        self.total_load = parameters.ems.gesamtlast[:time_steps]  # Required total energy
        self.pv_forecast = parameters.ems.pv_prognose_wh[:time_steps]  # Forecasted production

        # Price parameters
        p_import = parameters.ems.strompreis_euro_pro_wh[:time_steps]  # Price for buying from grid

        if isinstance(p_import, list):
            self.price_import = p_import
        else:
            self.price_import = [p_import] * len(self.total_load)

        p_export = parameters.ems.einspeiseverguetung_euro_pro_wh  # Price for selling to grid
        if isinstance(p_export, list):
            self.price_export = p_export
        else:
            self.price_export = [p_export] * len(self.total_load)

        self.price_storage = parameters.ems.preis_euro_pro_wh_akku
        # Value of stored energy at end of horizon

    def add_battery(self, parameters, batt_type: str):
        """Add battery parameters to the grid model.

        Parameters
        ----------
        parameters : object
            Object containing battery configurations
            Expected to have attributes like 'pv_akku' or 'eauto'
        batt_type : str
            Battery type identifier (e.g., 'pv_akku' or 'eauto')

        Returns:
        -------
        None
            This method modifies the current instance in-place

        Notes:
        -----
        If battery attributes are not found, default values are used:
        - min_soc_percentage: 0%
        - max_soc_percentage: 100%
        - init_soc_percentage: 50%
        - max_charge_power_w: 0W
        - capacity_wh: 0Wh
        - charging_efficiency: 1.0
        - discharging_efficiency: 1.0

        The method adds the battery type to the internal registry of
        available batteries if successful.
        """
        # Get battery configuration if it exists
        battery = getattr(parameters, batt_type, None)

        from akkudoktoreos.devices.battery import SolarPanelBatteryParameters

        if battery is not None:
            # Add all battery parameters with their default values if not specified
            self.soc_min[batt_type] = getattr(battery, "min_soc_percentage", 0)
            self.soc_max[batt_type] = getattr(battery, "max_soc_percentage", 100)
            self.soc_init[batt_type] = getattr(battery, "init_soc_percentage", 50)
            self.power_max[batt_type] = getattr(battery, "max_charge_power_w", 0)
            self.capacity[batt_type] = getattr(battery, "capacity_wh", 0)
            self.eff_charge[batt_type] = getattr(battery, "charging_efficiency", 1)
            self.eff_discharge[batt_type] = getattr(battery, "discharging_efficiency", 1)
            self.can_discharge[batt_type] = (
                True if isinstance(battery, SolarPanelBatteryParameters) else False
            )

            # Add battery type to the set of available batteries
            self.battery_set.append(batt_type)
