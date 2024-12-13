from src.data.utils.config import get_config
from src.data.base.progress_bar import ProgressBar
import yaml
import os

class ExogenousDataSelector:
    def __init__(self, exogenous_data):
        """
        a
        """

        self.config_dict = get_config("exog")
        self.yaml_path = os.path.join(os.getcwd(), "exog.config.yaml")
        self.exogenous_data = exogenous_data
        self.groups = self.config_dict["groups"]

        use_selected, vars_by_group = self.config_dict["selected_variables"], self.config_dict["exogenous_data"]

        if use_selected:
            print("Previous configuration found:")
            for group, variables in vars_by_group.items():
                print(f"- {group.capitalize()}: {', '.join(variables)}")

            
            use_existing = input("Do you want to use the existing configuration? (y/n): ").strip().lower()
            if use_existing == "y":
                self.selected_variables = [
                    var for var_in_group in vars_by_group.values() for var in var_in_group
                ]
                return
            
        self.selected_variables = None
        self.grouped_variables = {}
        print("\nSelect the variables you want to include:")
        for group, variables in self.groups.items():
            available = [var for var in variables if var in self.exogenous_data.columns]
            self.grouped_variables[group] = available
            print(f"- {group.capitalize()}: {', '.join(available)}")


    def save_config(self, selected_variables, grouped_data):
        """
        Save the selected variables to the configuration file (YAML)
        """

        if os.path.exists(self.yaml_path):
            with open(self.yaml_path, "r") as file:
                config = yaml.safe_load(file) or {}
        else:
            config = {}

        config["selected_variables"] = selected_variables
        config["exogenous_data"] = grouped_data

        with open(self.yaml_path, "w") as file:
            yaml.dump(config, file)
        print(f"\nConfiguración actualizada en {self.yaml_path}")

    def select(self, exogenous_data):
        """
        Select variables by group or individually
        """

        if not self.selected_variables:
            print("\nSelect variables by group or individually:")

            print("1. Select by group (e.g. macroeconomic, weather, discounts)")
            print("2. Select specific variables")
            print("3. Select all variables (ENTER)")

            user_input = input("Option: ").strip()

            if not user_input:
                self.selected_variables = self.exogenous_data.columns.to_list()
            elif user_input.isnumeric():
                self._select_specific_variables()
            else:
                self._select_groups(user_input)

            print(f"\nSelected variables: {self.selected_variables}")
            print("\nDo you want to save this configuration? (y/n)")
            save_choice = input().strip().lower()

            if save_choice == "y":
                grouped_data = {
                    group: [var for var in self.selected_variables if var in self.grouped_variables.get(group, [])]
                    for group in self.groups.keys()
                }
                self.save_config(selected_variables=True, grouped_data=grouped_data)
            else:
                self.save_config(selected_variables=False, grouped_data={})

        if "fecha" in self.exogenous_data.columns and "fecha" not in self.selected_variables:
            self.selected_variables.append("fecha")

        return self.exogenous_data[self.selected_variables]
    
    def _select_groups(self, group_input):
        """
        Select variables by group
        """
        groups_selected = [grp.strip() for grp in group_input.split(",")]
        self.selected_variables = []
        for group in groups_selected:
            if group in self.grouped_variables:
                self.selected_variables.extend(self.grouped_variables[group])
            else:
                print(f"Group '{group}' not found. Skipping...")

    def _select_specific_variables(self):
        """
        Select specific variables
        """
        print("\nSelect variables:")
        for i, var in enumerate(self.exogenous_data.columns):
            print(f"{i + 1}. {var}")
        user_input = input("Please enter index separated by commas (e.g., 1,2,3): ").strip()
        selected_indices = [int(i.strip()) - 1 for i in user_input.split(",")]
        self.selected_variables = [self.exogenous_data.columns[i] for i in selected_indices]
                
        
