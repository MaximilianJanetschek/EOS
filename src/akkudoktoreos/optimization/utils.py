from dataclasses import dataclass, field
from typing import Dict, List

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.optimization.genetic import OptimizationParameters


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
    price_storage: float = field(default_factory=float)
    can_discharge: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def init_from_parameters(
        cls, parameters: OptimizationParameters, config: ConfigEOS
    ) -> "ModelParameters":
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

    def add_ems_parameters(
        self, parameters: OptimizationParameters, time_steps: int | None
    ) -> None:
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
        if time_steps is None:
            time_steps = len(parameters.ems.gesamtlast)
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

    def add_battery(self, parameters: dict, batt_type: str) -> None:
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
