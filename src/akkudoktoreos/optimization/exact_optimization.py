from typing import Any, Optional
from pydantic import Field
from pyscipopt import Model, quicksum, Variable

from akkudoktoreos.core.coreabc import (
    ConfigMixin,
    DevicesMixin,
    EnergyManagementSystemMixin,
)
from akkudoktoreos.core.pydantic import ParametersBaseModel
from akkudoktoreos.optimization.genetic import OptimizationParameters
from akkudoktoreos.optimization.utils import visualize_warm_start, ModelParameters
from dataclasses import dataclass, field
from typing import Dict
from akkudoktoreos.optimization.greedy_construction import GreedyConstructionSolver


class ExactSolutionResponse(ParametersBaseModel):
    """Response model for the exact optimization solution."""

    akku_charge: list[float] = Field(
        description="Array with target charging / Discharging values in wh."
    )
    eauto_charge: Optional[list[float]] = Field(
        default=None,
        description="Array containing electric vehicle charging values in wh.",
    )


@dataclass
class ModelVariables:
    """The Model Variables dataclass stores all information about the decision varialbes of a optimization prolbem."""

    charge: Dict[tuple[str, int], Variable] = field(default_factory=dict)
    discharge: Dict[tuple[str, int], Variable] = field(default_factory=dict)
    soc: Dict[tuple[str, int], Variable] = field(default_factory=dict)
    grid_import: Dict[int, Variable] = field(default_factory=dict)
    grid_export: Dict[int, Variable] = field(default_factory=dict)
    flow_direction: Dict[int, Variable] = field(default_factory=dict)


class MILPOptimization(ConfigMixin, DevicesMixin, EnergyManagementSystemMixin):
    """Mixed-Integer Linear Programming Optimization for Energy Management Systems.

    This class implements a Mixed-Integer Linear Programming (MILP) formulation that
    minimizes energy costs while satisfying system constraints. It considers multiple
    energy sources and storage devices, including PV systems, batteries, and electric vehicles.

    The optimization problem is solved using the SCIP solver through the PySCIPOpt interface.

    Attributes:
        opti_param (Dict[str, Any]): Dictionary storing optimization parameters.
        possible_charge_values (List[float]): List of available charge rates as percentages.
        verbose (bool): Flag to control logging verbosity.
    """

    def __init__(
        self,
        verbose: bool = False,
    ):
        """Initialize the MILP optimization problem.

        Args:
            verbose (bool, optional): Enable verbose output. Defaults to False.
        """
        self.opti_param: dict[str, Any] = {}
        self.verbose = verbose

    def optimize_ems(self, parameters: OptimizationParameters) -> ExactSolutionResponse:
        """Solve the energy management system optimization problem using MILP.

        This method formulates and solves a MILP problem to minimize energy costs while satisfying
        system constraints. The optimization considers:
        - Grid power exchange (import/export)
        - Battery storage systems
        - PV generation
        - Electric vehicle charging
        - Time-varying electricity prices

        Args:
            parameters (OptimizationParameters): Input parameters containing:
                - Load profiles (total_load)
                - PV generation forecast (pv_forecast_wh)
                - Battery parameters (capacity, efficiency, power limits)
                - Price data (grid import/export prices)
                - Initial conditions

        Returns:
            ExactSolutionResponse: Optimization results containing optimal charging schedules.

        Raises:
            ValueError: If no optimal solution is found.

        Note:
            The optimization problem includes the following key components:

            Variables:
                - c[i,t]: Charging power for storage device i at time t
                - d[i,t]: Discharging power for storage device i at time t
                - s[i,t]: State of charge for storage device i at time t
                - n[t]: Grid import power at time t
                - e[t]: Grid export power at time t

            Constraints:
                1. Power balance at each timestep
                2. Battery dynamics (state of charge evolution)
                3. Operating limits (power, energy capacity)
                4. Grid power flow directionality

            Objective:
                Maximize: sum(-n[t]*p_N[t] + e[t]*p_E[t]) + sum(s[i,T]*p_a)
                where:
                - p_N: Grid import price
                - p_E: Grid export price
                - p_a: Final state of charge value
                - T: Final timestep
        """
        # Create optimization model
        model = Model("energy_management")

        # Define sets
        time_steps = range(self.config.optimization_hours)  # Time steps

        grid_model = ModelParameters.init_from_parameters(parameters, config=self.config)

        # Create variables
        charge = {}  # Charging power
        discharge = {}  # Discharging power
        soc = {}  # State of charge
        for batt_type in grid_model.battery_set:
            discharge_factor = 1 if grid_model.can_discharge[batt_type] else 0
            for t in time_steps:
                charge[batt_type, t] = model.addVar(
                    name=f"charge_{batt_type}_{t}",
                    vtype="C",
                    lb=0,
                    ub=grid_model.power_max[batt_type],
                )
                discharge[batt_type, t] = model.addVar(
                    name=f"discharge_{batt_type}_{t}",
                    vtype="C",
                    lb=0,
                    ub=grid_model.power_max[batt_type] * discharge_factor,
                )
                soc[batt_type, t] = model.addVar(
                    name=f"soc_{batt_type}_{t}",
                    vtype="C",
                    lb=grid_model.soc_min[batt_type],
                    ub=grid_model.soc_max[batt_type],
                )

        grid_import = {}  # Grid import power
        grid_export = {}  # Grid export power
        for t in time_steps:
            grid_import[t] = model.addVar(name=f"grid_import_{t}", vtype="C", lb=0)
            grid_export[t] = model.addVar(name=f"grid_export_{t}", vtype="C", lb=0)

        # Add constraints
        # Grid balance constraint
        for t in time_steps:
            model.addCons(
                quicksum(
                    -charge[batt_type, t] + discharge[batt_type, t]
                    for batt_type in grid_model.battery_set
                )
                + grid_model.pv_forecast[t]
                + grid_import[t]
                == grid_export[t] + grid_model.total_load[t],
                name=f"grid_balance_{t}",
            )

        # Battery dynamics constraints
        for batt_type in grid_model.battery_set:
            for t in time_steps:
                if t == time_steps[0]:
                    model.addCons(
                        grid_model.soc_init[batt_type] * grid_model.capacity[batt_type] / 100
                        + grid_model.eff_charge[batt_type] * charge[batt_type, t]
                        - (1 / grid_model.eff_discharge[batt_type]) * discharge[batt_type, t]
                        == soc[batt_type, t] * grid_model.capacity[batt_type] / 100,
                        name=f"battery_dynamics_{batt_type}_{t}",
                    )
                else:
                    model.addCons(
                        soc[batt_type, t - 1] * grid_model.capacity[batt_type] / 100
                        + grid_model.eff_charge[batt_type] * charge[batt_type, t]
                        - (1 / grid_model.eff_discharge[batt_type]) * discharge[batt_type, t]
                        == soc[batt_type, t] * grid_model.capacity[batt_type] / 100,
                        name=f"battery_dynamics_{batt_type}_{t}",
                    )

        # Prevent simultaneous import and export when import price is less than or equal to export price
        flow_var = {}
        for t in time_steps:
            if isinstance(grid_model.price_export, float):
                enforce_flow = grid_model.price_import[t] <= grid_model.price_export
            else:
                enforce_flow = grid_model.price_import[t] <= grid_model.price_export[t]

            if enforce_flow:
                flow_var[t] = model.addVar(name=f"flow_direction_{t}", vtype="B", lb=0, ub=1)
                max_bezug = sum(
                    grid_model.eff_charge[batt_type] * grid_model.power_max[batt_type]
                    for batt_type in grid_model.battery_set
                ) + max(grid_model.total_load)
                max_einspeise = sum(
                    grid_model.eff_discharge[batt_type] * grid_model.power_max[batt_type]
                    for batt_type in grid_model.battery_set
                ) + max(grid_model.pv_forecast)
                big_m = max(max_bezug, max_einspeise)
                model.addCons(grid_export[t] <= big_m * flow_var[t], name=f"export_constraint_{t}")
                model.addCons(
                    grid_import[t] <= big_m * (1 - flow_var[t]),
                    name=f"import_constraint_{t}",
                )

        # Set objective
        objective = quicksum(
            -grid_import[t] * grid_model.price_import[t]
            + grid_export[t] * grid_model.price_export[t]
            for t in time_steps
        ) + quicksum(
            soc[batt_type, time_steps[-1]]
            * grid_model.price_storage
            * grid_model.capacity[batt_type]
            for batt_type in grid_model.battery_set
        )
        model.setObjective(objective, "maximize")

        model_vars = ModelVariables(
            charge=charge,
            discharge=discharge,
            soc=soc,
            grid_import=grid_import,
            grid_export=grid_export,
            flow_direction=flow_var,
        )

        # set warm start
        self.set_warm_start(model, model_vars, time_steps, grid_model)

        model.optimize()

        # Solve the model
        if self.verbose:
            print("Number of variables:", len(model.getVars()))
            print("Number of constraints:", len(model.getConss()))
            print("Objective value:", model.getObjVal())

        if model.getStatus() != "optimal":
            raise ValueError("No optimal solution found")

        # Extract solution
        if "pv_akku" in grid_model.battery_set:
            akku_charge = [
                model.getVal(charge["pv_akku", t]) - model.getVal(discharge["pv_akku", t])
                for t in time_steps
            ]
            for i in time_steps:
                print(
                    model.getVal(soc["pv_akku", i]),
                    model.getVal(charge["pv_akku", i]),
                    model.getVal(discharge["pv_akku", i]),
                )
        else:
            akku_charge = []

        if "eauto" in grid_model.battery_set:
            ev_charge = [model.getVal(charge["eauto", t]) for t in time_steps]
        else:
            ev_charge = None

        return ExactSolutionResponse(
            akku_charge=akku_charge,
            eauto_charge=ev_charge,
        )

    def set_warm_start(
        self,
        model: Model,
        vars: ModelVariables,
        time_steps: range,
        model_params: ModelParameters,
    ):
        # Calculate warm start solution
        greedy_solver = GreedyConstructionSolver(verbose=self.verbose)
        greedy_start = greedy_solver.generate_warm_start(
            time_steps=time_steps, model_params=model_params
        )

        if False:
            visualize_warm_start(
                heur_sol=greedy_start, model_params=model_params, time_steps=time_steps
            )

        # Create a solution object
        solution = model.createSol()

        # Set variable values in the solution object
        for batt_idx, batt_type in greedy_start.battery_dict.items():
            for t in time_steps:
                # Set battery-related variables
                model.setSolVal(
                    solution,
                    vars.charge[batt_type, t],
                    greedy_start.charge[batt_idx, t],
                )
                model.setSolVal(
                    solution,
                    vars.discharge[batt_type, t],
                    greedy_start.discharge[batt_idx, t],
                )
                model.setSolVal(solution, vars.soc[batt_type, t], greedy_start.soc[batt_idx, t])

        # Set grid import/export values
        for t in time_steps:
            model.setSolVal(solution, vars.grid_import[t], greedy_start.grid_import[t])
            model.setSolVal(solution, vars.grid_export[t], greedy_start.grid_export[t])

            # Set flow direction binary variables if they exist
            if t in vars.flow_direction.keys():
                model.setSolVal(solution, vars.flow_direction[t], greedy_start.flow_direction[t])

        # Try to add the solution to the model

        try:
            accepted = model.checkSol(solution, completely=True, original=True)

            if accepted:
                try:
                    model.addSol(solution)
                    print("Solution successfully added")
                except Exception as e:
                    raise ValueError
                    print(f"Error adding solution: {e}")
            else:
                raise ValueError(f"Solution was not accepted: {accepted}. Check feasability!")

        except Exception as e:
            print(f"Error checking solution: {e}")
