import time
from dataclasses import dataclass

import numpy as np

from akkudoktoreos.optimization.utils import ModelParameters


@dataclass
class HeuristicSolution:
    """Stores heuristic solutions to mixed-integer optimization problems.

    This dataclass maintains a warm-start solution for mixed-integer optimization
    problems in energy management systems. It provides a foundation for reducing the
    lower bound of the optimization problem, which helps in two ways:

    1. Reduces problem complexity through extended information
    2. Enables warm-starting the optimization process to avoid unnecessary steps

    Attributes:
    ----------
    battery_dict : dict[int, str]
        Maps array indices to battery identifiers, where keys are positions
        in the array and values are battery names
    charge : np.ndarray
        2D array of charging power values, shape (num_batteries, num_timesteps)
    discharge : np.ndarray
        2D array of discharging power values, shape (num_batteries, num_timesteps)
    soc : np.ndarray
        2D array of state of charge values, shape (num_batteries, num_timesteps)
    grid_import : np.ndarray
        1D array of grid import values for each timestep
    grid_export : np.ndarray
        1D array of grid export values for each timestep
    flow_direction : np.ndarray
        1D array indicating power flow direction (0: import, 1: export)
    """

    battery_dict: dict[int, str]  # determines position in array, [battery_pos,t]
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    grid_import: np.ndarray
    grid_export: np.ndarray
    flow_direction: np.ndarray

    @classmethod
    def from_params(cls, model_params: ModelParameters, time_steps: range):
        """Create a new HeuristicSolution from model parameters.

        Initializes a solution instance with default values based on the provided
        model parameters, setting up the necessary arrays with appropriate dimensions
        and initializing state of charge values.

        Parameters
        ----------
        model_params : ModelParameters
            Model parameters containing battery configurations and system constraints
        time_steps : range
            Range object defining the optimization time horizon

        Returns:
        -------
        HeuristicSolution
            Initialized solution object with default values

        Notes:
        -----
        The method initializes:
        - Battery dictionary mapping positions to battery names
        - Zero-filled arrays for charge/discharge/grid interactions
        - SoC array initialized to each battery's specified initial SoC
        """
        battery_dict = {i: batt for i, batt in enumerate(model_params.battery_set)}
        charge = np.zeros((len(model_params.battery_set), len(time_steps)))
        discharge = np.zeros((len(model_params.battery_set), len(time_steps)))
        soc = np.ones((len(battery_dict), len(time_steps)))

        # set both battery
        for idx, bat in battery_dict.items():
            soc[idx, :] *= model_params.soc_init[bat]

        grid_import = np.zeros(len(time_steps))
        grid_export = np.zeros(len(time_steps))
        flow_direction = np.zeros(len(time_steps))

        return cls(
            battery_dict,
            charge,
            discharge,
            soc,
            grid_import,
            grid_export,
            flow_direction,
        )

    def _update_grid_values(self, model_params: ModelParameters, time_steps: range):
        """Calculate grid interactions based on power balance across the system.

        This method vectorizes the grid interaction calculations for the entire time horizon,
        determining whether each timestep requires power import from the grid or allows
        export to the grid based on the overall power balance.

        Parameters
        ----------
        model_params : ModelParameters
            Model parameters containing PV forecast and load data

        Returns
        -------
        None
            Updates grid_import, grid_export, and flow_direction attributes in-place

        Notes
        -----
        The power balance calculation follows these steps:
        1. Compute net battery power (sum of discharge minus charge across all batteries)
        2. Calculate system power balance: PV generation + battery power - load demand
        3. Determine flow direction (import when net_power < 0, export when net_power >= 0)
        4. Set grid_import values where power deficit exists (negative net_power)
        5. Set grid_export values where power surplus exists (positive net_power)

        Flow direction is represented as a binary array:
        - 0: importing power from the grid
        - 1: exporting power to the grid
        """

        # Calculate power balance across the system
        battery_net_power = np.sum(self.discharge - self.charge, axis=0)
        net_power = model_params.pv_forecast + battery_net_power - model_params.total_load

        # Create mask for import/export conditions
        import_mask = net_power < 0
        export_mask = ~import_mask

        # Initialize grid interaction arrays
        self.grid_import = np.zeros_like(net_power)
        self.grid_export = np.zeros_like(net_power)

        # Set values based on power flow direction
        self.grid_import[import_mask] = -net_power[import_mask]  # Convert negative to positive
        self.grid_export[export_mask] = net_power[export_mask]

        # Update flow direction (0: import, 1: export)
        self.flow_direction = export_mask.astype(int)



class GreedyConstructionSolver:
    """Implements a greedy construction algorithm for energy management system optimization.

    This solver creates warm-start solutions for mixed-integer linear programming (MILP)
    optimization problems by using a multi-pass greedy algorithm. It prioritizes charging
    batteries with excess solar power, ensures minimum state of charge constraints are met,
    and optimizes for price arbitrage opportunities.

    Attributes:
    ----------
    verbose : bool
        Whether to print debug information during the solving process
    time_steps : range
        Range of optimization timesteps
    model_params : ModelParameters
        Model parameters containing battery configurations and system constraints
    """

    verbose: bool
    time_steps: range
    model_params: ModelParameters

    def __init__(self, verbose: bool = False):
        """Initialize the GreedyConstructionSolver.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print debug information during the solving process, by default False
        """
        self.verbose = verbose

    def generate_warm_start(
        self,
        time_steps: range,
        model_params: ModelParameters,
    ) -> HeuristicSolution:
        """Generate improved warm start solution for the MILP optimization.

        This implementation follows five main passes:

        1. First pass: Prioritize charging batteries with excess solar power up to 90% SoC
        2. Second pass: Ensure minimum SoC requirements by charging at lowest grid import cost
        3. Third pass: Optimize by using excess battery SoC during high price periods
        4. Fourth pass: Perform time-based arbitrage for the first battery
        5. Fifth pass: Validate the solution to ensure all constraints are met

        Parameters
        ----------
        time_steps : range
            Range of optimization timesteps
        model_params : ModelParameters
            Model parameters containing load profiles, PV forecast, battery configurations,
            pricing data, and system constraints

        Returns:
        -------
        HeuristicSolution
            Optimized solution containing arrays for:
            - battery_dict: Mapping between array indices and battery identifiers
            - charge: Charging power for each battery at each timestep
            - discharge: Discharging power for each battery at each timestep
            - soc: State of charge (as percentage 0-100) for each battery at each timestep
            - grid_import: Grid import power at each timestep
            - grid_export: Grid export power at each timestep
            - flow_direction: Binary variables indicating flow direction (0=import, 1=export)
        """
        # Track overall start time
        overall_start = time.time()

        # save variables
        self.model_params = model_params
        self.time_steps = time_steps

        # Initialize solution dictionaries
        greedy_sol = HeuristicSolution.from_params(model_params=model_params, time_steps=time_steps)

        # ----- FIRST PASS: Prioritized charging with excess solar, starting from first battery -----
        greedy_sol = self.construct_greedy_solution(greedy_sol)

        # ----- SECOND PASS: Ensure minimum SOC at EVERY time step -----
        feasible_sol = self.greedy_to_feasible(greedy_sol)

        # ------ THIRD PASS: Use the excess battery SoC in times where prices are super high
        improved_sol = self.improve_battery_usage(feasible_sol)

        # ----- FOURTH PASS: Price optimization for the first battery -----
        final_sol = self.time_swap(improved_sol)

        # ----- FIFTH PASS: Validation of input -----
        final_sol = self.ajdust_values(final_sol)

        if self.verbose:
            # Calculate total time
            total_time = time.time() - overall_start

            # Calculate and print percentages
            print(f"Total time:  {total_time:.4f} seconds")

        return final_sol

    def ajdust_values(self, greedy_sol: HeuristicSolution) -> HeuristicSolution:
        # make sure params are correct based on battery params
        for batt_idx, batt_type in greedy_sol.battery_dict.items():
            soc_change =  (
                    self.model_params.eff_charge[batt_type] * greedy_sol.charge[batt_idx, :]
                    -  greedy_sol.discharge[batt_idx, :] / self.model_params.eff_discharge[batt_type]
                ) / self.model_params.capacity[batt_type] *100

            cumsum_change = np.cumsum(soc_change)
            greedy_sol.soc[batt_idx, :] = self.model_params.soc_init[batt_type]  + cumsum_change


        greedy_sol._update_grid_values(model_params=self.model_params, time_steps=self.time_steps)

        return greedy_sol



    def validate_sol(self, greedy_sol: HeuristicSolution) -> HeuristicSolution:
        """Validate and correct the solution to ensure all constraints are met.

        This method performs a final validation pass to ensure the solution meets
        all battery constraints, particularly the minimum and maximum state of charge
        limits at each timestep.

        Parameters
        ----------
        greedy_sol : HeuristicSolution
            The solution to validate and correct

        Returns:
        -------
        HeuristicSolution
            The validated and corrected solution

        Notes:
        -----
        The method checks each battery's state of charge trajectory and makes
        adjustments to charging and discharging values if constraints are violated.
        It focuses on:

        1. Enforcing minimum SoC constraint at the last timestep
        2. Preventing exceeding maximum SoC at any timestep
        3. Recalculating grid import/export values after corrections
        """

        last_timestep = self.time_steps[-1]
        for batt_idx, batt_type in greedy_sol.battery_dict.items():
            current_soc_pct = self.model_params.soc_init[batt_type]

            for t in self.time_steps:
                # Calculate SoC change from charge/discharge
                energy_gained = (
                    greedy_sol.charge[batt_idx, t] * self.model_params.eff_charge[batt_type]
                )
                energy_lost = (
                    greedy_sol.discharge[batt_idx, t] / self.model_params.eff_discharge[batt_type]
                )
                soc_change_pct = (
                    (energy_gained - energy_lost) / self.model_params.capacity[batt_type]
                ) * 100

                # Check if next SoC would be valid
                next_soc_pct = current_soc_pct + soc_change_pct

                # Only enforce minimum SoC constraint at the last timestep
                if t == last_timestep and next_soc_pct < self.model_params.soc_min[batt_type]:
                    # Adjust charging/discharging to meet minimum SoC
                    if greedy_sol.discharge[batt_idx, t] > 0:
                        # First try reducing discharge
                        discharge_reduction = min(
                            greedy_sol.discharge[
                                batt_idx, t
                            ],  # Cannot reduce more than current discharge
                            (self.model_params.soc_min[batt_type] - next_soc_pct)
                            * self.model_params.capacity[batt_type]
                            / 100
                            * self.model_params.eff_discharge[
                                batt_type
                            ],  # Energy needed to meet min SoC
                        )

                        greedy_sol.discharge[batt_idx, t] -= discharge_reduction

                        # Recalculate next SoC
                        energy_gained = (
                            greedy_sol.charge[batt_idx, t] * self.model_params.eff_charge[batt_type]
                        )
                        energy_lost = (
                            greedy_sol.discharge[batt_idx, t]
                            / self.model_params.eff_discharge[batt_type]
                        )
                        soc_change_pct = (
                            (energy_gained - energy_lost) / self.model_params.capacity[batt_type]
                        ) * 100
                        next_soc_pct = current_soc_pct + soc_change_pct

                    if next_soc_pct < self.model_params.soc_min[batt_type]:
                        # If still below min, increase charging
                        shortfall_pct = self.model_params.soc_min[batt_type] - next_soc_pct
                        shortfall_energy = (
                            shortfall_pct * self.model_params.capacity[batt_type]
                        ) / 100
                        additional_charge = (
                            shortfall_energy / self.model_params.eff_charge[batt_type]
                        )

                        # Limit by maximum power
                        additional_charge = min(
                            additional_charge,
                            self.model_params.power_max[batt_type]
                            - greedy_sol.charge[batt_idx, t],  # Remaining charge capacity
                        )

                        greedy_sol.charge[batt_idx, t] += additional_charge

                        # Final recalculation
                        energy_gained = (
                            greedy_sol.charge[batt_idx, t] * self.model_params.eff_charge[batt_type]
                        )
                        energy_lost = (
                            greedy_sol.discharge[batt_idx, t]
                            / self.model_params.eff_discharge[batt_type]
                        )
                        soc_change_pct = (
                            (energy_gained - energy_lost) / self.model_params.capacity[batt_type]
                        ) * 100
                        next_soc_pct = current_soc_pct + soc_change_pct

                # Also check for exceeding maximum SoC
                if next_soc_pct > self.model_params.soc_max[batt_type]:
                    # First try reducing charging
                    if greedy_sol.charge[batt_idx, t] > 0:
                        charge_reduction = min(
                            greedy_sol.charge[
                                batt_idx, t
                            ],  # Cannot reduce more than current charge
                            (next_soc_pct - self.model_params.soc_max[batt_type])
                            * self.model_params.capacity[batt_type]
                            / 100
                            / self.model_params.eff_charge[
                                batt_type
                            ],  # Excess energy causing overfill
                        )

                        greedy_sol.charge[batt_idx, t] -= charge_reduction

                        # Recalculate next SoC
                        energy_gained = (
                            greedy_sol.charge[batt_idx, t] * self.model_params.eff_charge[batt_type]
                        )
                        energy_lost = (
                            greedy_sol.discharge[batt_idx, t]
                            / self.model_params.eff_discharge[batt_type]
                        )
                        soc_change_pct = (
                            (energy_gained - energy_lost) / self.model_params.capacity[batt_type]
                        ) * 100
                        next_soc_pct = current_soc_pct + soc_change_pct

                    # If still above max, increase discharging
                    if next_soc_pct > self.model_params.soc_max[batt_type]:
                        excess_pct = next_soc_pct - self.model_params.soc_max[batt_type]
                        excess_energy = (excess_pct * self.model_params.capacity[batt_type]) / 100
                        additional_discharge = (
                            excess_energy * self.model_params.eff_discharge[batt_type]
                        )

                        # Limit by maximum power
                        additional_discharge = min(
                            additional_discharge,
                            self.model_params.power_max[batt_type]
                            - greedy_sol.discharge[batt_idx, t],
                            # Remaining discharge capacity
                        )

                        greedy_sol.discharge[batt_idx, t] += additional_discharge

                        # Final recalculation
                        energy_gained = (
                            greedy_sol.charge[batt_idx, t] * self.model_params.eff_charge[batt_type]
                        )
                        energy_lost = (
                            greedy_sol.discharge[batt_idx, t]
                            / self.model_params.eff_discharge[batt_type]
                        )
                        soc_change_pct = (
                            (energy_gained - energy_lost) / self.model_params.capacity[batt_type]
                        ) * 100
                        next_soc_pct = current_soc_pct + soc_change_pct

                # Update SOC for this timestep
                greedy_sol.soc[batt_idx, t] = next_soc_pct
                current_soc_pct = next_soc_pct

        # Final update of grid import/export
        greedy_sol._update_grid_values(model_params= self.model_params, time_steps= self.time_steps)

        return greedy_sol

    def time_swap(self, greedy_sol) -> HeuristicSolution:
        """Optimize the solution through time-based price arbitrage.

        This method identifies opportunities to charge batteries during low-price
        periods and discharge during high-price periods to maximize economic benefit.
        The method primarily focuses on the first battery in the set, assuming it is
        capable of both charging and discharging.

        Parameters
        ----------
        greedy_sol : HeuristicSolution
            The solution to optimize through time swapping

        Returns:
        -------
        HeuristicSolution
            The optimized solution with improved price arbitrage

        Notes:
        -----
        The method:
        1. Sorts timesteps by import prices
        2. For high-price periods with grid imports, looks for earlier low-price periods
           where charging would be profitable
        3. Evaluates profitability of charge/discharge cycles accounting for efficiency losses
        4. Adjusts charging and discharging patterns to maximize economic value
        """
        # Get only the import prices for our time steps
        import_prices_array = np.array([self.model_params.price_import[t] for t in self.time_steps])

        # Assume the first battery in the list is capable of both charging and discharging
        # This pass is only applicable if we have at least one battery
        if self.model_params.battery_set:
            # Get the first battery (assuming it's the main battery that can both charge and discharge)
            # todo set battery on
            main_battery = self.model_params.battery_set[0]
            batt_idx = 0

            # Create a list of timesteps with grid import costs
            low_price_times = np.argsort(import_prices_array)
            high_price_times = low_price_times[::-1]

            soc = greedy_sol.soc[batt_idx, :]

            import_times = np.where(greedy_sol.grid_import > 0)[0]
            cand = np.intersect1d(high_price_times, import_times)

            def find_first_position(arr: np.ndarray, value: float) -> int:
                indices = np.where(arr == value)[0]
                if len(indices) > 0:
                    return indices[-1]
                else:
                    return -1  # or None, or raise an exception

            # For each high price time where we're importing from grid
            for high_idx in cand:
                high_price = import_prices_array[high_idx]

                # Calculate maximum discharge potential at this timestep
                current_battery_discharge = greedy_sol.discharge[batt_idx, high_idx]
                additional_discharge_power = min(
                    self.model_params.power_max[main_battery]
                    - current_battery_discharge,  # Power limit
                    greedy_sol.grid_import[
                        high_idx
                    ],  # Only discharge up to the current grid import amount
                )

                if additional_discharge_power <= 0:
                    continue  # No additional discharge possible

                # check if any time is at max before
                max_soc_idx = find_first_position(
                    greedy_sol.soc[batt_idx, :], self.model_params.soc_max[main_battery]
                )
                # Calculate how much energy would be needed for the discharge, accounting for efficiency
                energy_needed = (
                    additional_discharge_power / self.model_params.eff_discharge[main_battery]
                )
                charging_power_needed = energy_needed / self.model_params.eff_charge[main_battery]

                for low_idx in low_price_times[
                    (low_price_times < high_idx) & (low_price_times > max_soc_idx)
                ]:
                    # Check if we have capacity to charge at this time
                    available_charge_capacity = (
                        self.model_params.power_max[main_battery] - greedy_sol.charge[batt_idx, low_idx]
                    )

                    # Calculate actual charging power we can add
                    max_pos_soc = self.model_params.soc_max[main_battery] - np.max(
                        greedy_sol.soc[batt_idx, low_idx:high_idx]
                    )
                    max_soc_charge = (
                        max_pos_soc * self.model_params.capacity[main_battery] / 100
                    ) / self.model_params.eff_charge[main_battery]

                    charge_power_to_add = min(
                        charging_power_needed, available_charge_capacity, max_soc_charge
                    )

                    if charge_power_to_add <= 0:
                        # if not available_charge_capacity was limiting factor we can stop here as no other will be better
                        if not charge_power_to_add == available_charge_capacity:
                            break
                        else:
                            continue

                    low_price = import_prices_array[low_idx]
                    # Calculate how much we can actually discharge with this amount of charge
                    discharge_power_possible = (
                        charge_power_to_add
                        * self.model_params.eff_charge[main_battery]
                        * self.model_params.eff_discharge[main_battery]
                    )
                    soc_change = (
                        charge_power_to_add
                        * self.model_params.eff_charge[main_battery]
                        / self.model_params.capacity[main_battery]
                        * 100
                    )

                    # Check if this arbitrage would be profitable
                    cost_to_charge = charge_power_to_add * low_price
                    savings_from_discharge = discharge_power_possible * high_price

                    if savings_from_discharge < cost_to_charge:
                        continue  # Not profitable

                    # change soc, change charge and discharge
                    greedy_sol.charge[batt_idx, low_idx] += (
                        charge_power_to_add
                    )
                    greedy_sol.discharge[batt_idx, high_idx] += (
                        discharge_power_possible
                    )
                    greedy_sol.grid_import[high_idx] -= discharge_power_possible
                    greedy_sol.grid_import[low_idx] += charge_power_to_add
                    greedy_sol.soc[batt_idx, low_idx:high_idx] += soc_change
                    break  # Found a charging time for this discharge opportunity

        return greedy_sol

    def improve_battery_usage(self, greedy_sol: HeuristicSolution) -> HeuristicSolution:
        """Improve battery usage by utilizing excess capacity during high-price periods.

        This method identifies batteries with excess state of charge at the end of the
        optimization horizon and attempts to use this excess capacity during high-price
        periods to reduce grid imports.

        Parameters
        ----------
        greedy_sol : HeuristicSolution
            The solution to improve

        Returns:
        -------
        HeuristicSolution
            The solution with improved battery usage

        Notes:
        -----
        The method:
        1. Sorts timesteps by price (highest first)
        2. For timesteps with grid imports, checks if batteries have excess SoC at the end
        3. Calculates potential discharge without violating minimum SoC constraints
        4. Adjusts discharge values to reduce high-cost grid imports
        5. Recalculates SoC and grid import/export values

        The method skips electric vehicle batteries (if present) to preserve their charge.
        """
        # Get time indices sorted by price (highest first)
        # Only use indices that are within our time_steps
        prices_in_range = np.array(self.model_params.price_import[: len(self.time_steps)])
        des_prices = np.argsort(prices_in_range)[::-1]

        # Process high-price times first
        for idx in des_prices:
            t = idx

            # Check if we are importing
            if greedy_sol.grid_import[t] > 0:

                # Check if we have excess battery capacity at the end (above min_soc)
                for batt_idx, batt_type in greedy_sol.battery_dict.items():
                    # Skip electric vehicle battery if it exists
                    if batt_type == "eauto":
                        continue

                    # Check if we have excess SoC at the end
                    if (
                        greedy_sol.soc[batt_idx, self.time_steps[-1]]
                        > self.model_params.soc_min[batt_type]
                    ):

                        # Calculate how much we can discharge without violating min SoC
                        excess_soc_pct = (
                            greedy_sol.soc[batt_idx, self.time_steps[-1]]
                            - self.model_params.soc_min[batt_type]
                        )
                        excess_energy_wh = (
                            excess_soc_pct * self.model_params.capacity[batt_type]
                        ) / 100

                        # Convert to potential discharge power (accounting for efficiency)
                        potential_discharge = (
                            excess_energy_wh * self.model_params.eff_discharge[batt_type]
                        )

                        # Limit by maximum discharge power, available excess, and current grid import
                        available_discharge_power = min(
                            self.model_params.power_max[batt_type]
                            - greedy_sol.discharge[batt_idx, t],  # Power limit
                            potential_discharge,  # Energy from excess SoC
                            greedy_sol.grid_import[t],  # Don't discharge more than we're importing
                        )

                        if available_discharge_power > 0:
                            # Add discharge at this timestep
                            greedy_sol.discharge[batt_idx, t] += available_discharge_power

                            # update discharge in idx and reduce soc
                            soc_change_pct = (
                                available_discharge_power / self.model_params.capacity[batt_type] /  self.model_params.eff_discharge[batt_type]
                            ) * 100
                            greedy_sol.soc[batt_idx, t:] -= soc_change_pct

        # Update grid import/export after this change
        greedy_sol._update_grid_values(model_params = self.model_params, time_steps = self.time_steps)

        return greedy_sol

    def greedy_to_feasible(self, greedy_sol: HeuristicSolution) -> HeuristicSolution:
        """Convert a greedy solution into a feasible one by ensuring minimum SoC requirements.

        This method examines the solution for any violations of minimum state of charge
        constraints and makes adjustments to ensure feasibility by:
        1. Charging more during low-price periods
        2. Reducing discharge during earlier periods if necessary

        Parameters
        ----------
        greedy_sol : HeuristicSolution
            The greedy solution to make feasible

        Returns:
        -------
        HeuristicSolution
            A feasible solution that satisfies minimum SoC constraints

        Notes:
        -----
        For each battery, the method:
        1. Identifies timesteps where minimum SoC is violated
        2. Calculates the energy shortfall needed to meet the minimum SoC
        3. Attempts to charge more during earlier, low-price timesteps
        4. If charging is insufficient, reduces discharge in earlier timesteps
        5. Recalculates the complete SoC profile for all timesteps
        6. Updates grid import/export values based on the changes

        If minimum SoC constraints cannot be met, warning messages are printed.
        """

        # For each battery, check if minimum SoC is met at all time steps
        for batt_idx, batt_type in greedy_sol.battery_dict.items():
            # Second pass: Find violations and fix them
            for t in self.time_steps:
                # Check if SoC violates minimum requirement
                if greedy_sol.soc[batt_idx, t] < self.model_params.soc_min[batt_type]:
                    # Calculate shortfall
                    shortfall_pct = (
                        self.model_params.soc_min[batt_type] - greedy_sol.soc[batt_idx, t]
                    )
                    # Convert to energy (Wh)
                    shortfall_energy = (shortfall_pct * self.model_params.capacity[batt_type]) / 100

                    # Try to charge at the lowest price timesteps before this timestep
                    remaining_shortfall = shortfall_energy

                    # Get earlier time steps and their prices
                    earlier_indices = [i for i, t_val in enumerate(self.time_steps) if t_val < t]
                    if not earlier_indices:
                        continue  # No earlier times available

                    # Get prices for these earlier times
                    earlier_prices = [
                        self.model_params.price_import[self.time_steps[i]] for i in earlier_indices
                    ]
                    # Sort indices by price
                    price_order = np.argsort(earlier_prices)
                    cheapest_indices = [earlier_indices[i] for i in price_order]

                    for earlier_idx in cheapest_indices:
                        earlier_t = self.time_steps[earlier_idx]
                        # Calculate how much more we can charge at this timestep
                        available_charge_power = (
                            self.model_params.power_max[batt_type]
                            - greedy_sol.charge[batt_type, earlier_t]
                        )

                        if available_charge_power <= 0:
                            continue  # Already charging at maximum power

                        # Calculate energy we can gain with efficiency
                        max_energy_gain = (
                            available_charge_power * self.model_params.eff_charge[batt_type]
                        )

                        # Limit by the remaining shortfall
                        energy_to_add = min(max_energy_gain, remaining_shortfall)
                        power_to_add = energy_to_add / self.model_params.eff_charge[batt_type]

                        if power_to_add > 0:
                            # Add charge
                            greedy_sol.charge[batt_type, earlier_t] += power_to_add

                            # we need to check if the soc allows to transfer the power
                            soc_increase = energy_to_add / self.model_params.capacity[batt_type]

                            for t_test in range(earlier_t, t):
                                enough_gap = (
                                    greedy_sol.soc[batt_idx, t_test] + soc_increase
                                    <= self.model_params.soc_max[batt_type]
                                )
                                if not enough_gap:
                                    # there is not enough energy
                                    break

                            # Reduce remaining shortfall
                            remaining_shortfall -= energy_to_add

                            # If shortfall is eliminated, break
                            if remaining_shortfall <= 0:
                                break

                    # If we couldn't eliminate the shortfall by charging in earlier timesteps,
                    # we need to adjust the discharge decisions at earlier timesteps
                    if remaining_shortfall > 0:
                        for earlier_t in range(0, t):
                            # Calculate how much we can reduce discharge
                            reducible_discharge = greedy_sol.discharge[batt_type, earlier_t]

                            if reducible_discharge <= 0:
                                continue  # No discharge to reduce

                            # Calculate energy we can save by reducing discharge (accounting for efficiency)
                            max_energy_save = (
                                reducible_discharge / self.model_params.eff_discharge[batt_type]
                            )

                            # Limit by the remaining shortfall
                            energy_to_save = min(max_energy_save, remaining_shortfall)
                            discharge_to_reduce = (
                                energy_to_save * self.model_params.eff_discharge[batt_type]
                            )

                            if discharge_to_reduce > 0:
                                # Reduce discharge
                                greedy_sol.discharge[batt_type, earlier_t] -= discharge_to_reduce

                                # Reduce remaining shortfall
                                remaining_shortfall -= energy_to_save

                                # If shortfall is eliminated, break
                                if remaining_shortfall <= 0:
                                    break

            # Final pass: Recalculate SoC profile for all timesteps after modifications
            current_soc_pct = self.model_params.soc_init[batt_type]

            for t in self.time_steps:
                # Calculate energy change (in Wh)
                energy_gained = (
                    greedy_sol.charge[batt_idx, t] * self.model_params.eff_charge[batt_type]
                )
                energy_lost = (
                    greedy_sol.discharge[batt_idx, t] / self.model_params.eff_discharge[batt_type]
                )

                # Update SoC percentage
                soc_change_pct = (
                    (energy_gained - energy_lost) / self.model_params.capacity[batt_type]
                ) * 100
                current_soc_pct += soc_change_pct

                # Update SoC for this timestep
                greedy_sol.soc[batt_idx, t] = current_soc_pct

                # Double-check that minimum SOC is now met
                if greedy_sol.soc[batt_idx, t] < self.model_params.soc_min[batt_type]:
                    print(
                        f"Could not meet minimum SOC for battery {batt_type} at timestep {t}. "
                        f"Current: {greedy_sol.soc[batt_idx, t]:.2f}%, Minimum: {self.model_params.soc_min[batt_type]:.2f}%"
                    )

        # Update grid import/export after second pass
        greedy_sol._update_grid_values(model_params= self.model_params, time_steps=self.time_steps)

        return greedy_sol

    def construct_greedy_solution(self, greedy_sol: HeuristicSolution) -> HeuristicSolution:
        """Construct an initial greedy solution by prioritizing solar self-consumption.

        This method implements the first pass of the algorithm, which prioritizes
        charging batteries with excess solar power to increase self-consumption
        and reduce grid exports.

        Parameters
        ----------
        greedy_sol : HeuristicSolution
            An initialized solution with default values

        Returns:
        -------
        HeuristicSolution
            The solution after applying the greedy construction algorithm

        Notes:
        -----
        The method:
        1. Processes each timestep sequentially
        2. Calculates power balance (PV generation minus load)
        3. If excess power is available, charges batteries up to 90% SoC
        4. Respects maximum charging power and SoC constraints
        5. Updates SoC values and grid import/export for each timestep

        The method prioritizes charging to 90% rather than 100% to allow room
        for further optimization in later passes.
        """
        # Initialize current SoC for all batteries
        current_soc = {
            batt_type: self.model_params.soc_init[batt_type]
            for batt_type in self.model_params.battery_set
        }

        # Process each timestep
        for t in self.time_steps:
            # Calculate initial power balance (positive means excess PV)
            remaining_power = self.model_params.pv_forecast[t] - self.model_params.total_load[t]

            # If excess PV available, try to charge batteries starting from the last one
            if remaining_power > 0:
                for batt_idx, batt_type in greedy_sol.battery_dict.items():
                    # Skip if battery is already at or above min SoC
                    if current_soc[batt_type] >= 90:
                        continue

                    # Calculate maximum charging power considering all constraints
                    max_charge = min(
                        self.model_params.power_max[batt_type],  # Power limit
                        (self.model_params.soc_max[batt_type] - current_soc[batt_type])
                        * self.model_params.capacity[batt_type]
                        / self.model_params.eff_charge[batt_type],
                        remaining_power,  # Available PV excess
                    )

                    if max_charge > 0:
                        # Set the charge for this battery at this timestep
                        greedy_sol.charge[batt_idx, t] = max_charge

                        # Calculate energy gained (in Wh)
                        energy_gained = max_charge * self.model_params.eff_charge[batt_type]

                        # Update SoC percentage
                        soc_gained_pct = (
                            energy_gained / self.model_params.capacity[batt_type]
                        ) * 100
                        current_soc[batt_type] += soc_gained_pct

                        # Update SoC for this timestep
                        greedy_sol.soc[batt_idx, t] = current_soc[batt_type]

                        # Reduce remaining power
                        remaining_power -= max_charge

                        # If no more power to allocate, exit loop
                        if remaining_power <= 0:
                            break
            else:
                for batt_idx, batt_type in greedy_sol.battery_dict.items():
                    greedy_sol.soc[batt_idx, t] = current_soc[batt_type]

        # Update grid import/export after first pass
        greedy_sol._update_grid_values(model_params = self.model_params, time_steps= self.time_steps)

        return greedy_sol
