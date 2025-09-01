import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import io
import requests
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from final_model import DataFetcher, Main


class MLBPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MLB Win Prediction Visualizer")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.bg_color = "#f0f0f0"
        self.header_color = "#162B4B"  # MLB blue
        self.accent_color = "#D50032"  # MLB red
        self.text_color = "#333333"
        
        self.root.configure(bg=self.bg_color)
        
        # Configure styles
        self.style.configure('Header.TLabel', background=self.header_color, foreground='white', font=('Arial', 16, 'bold'), padding=10)
        self.style.configure('Title.TLabel', background=self.bg_color, foreground=self.header_color, font=('Arial', 14, 'bold'), padding=5)
        self.style.configure('Data.TLabel', background=self.bg_color, foreground=self.text_color, font=('Arial', 11))
        self.style.configure('Accent.TButton', background=self.accent_color, foreground='white', font=('Arial', 12, 'bold'))
        self.style.map('Accent.TButton', background=[('active', '#B5002B')])
        
        # Initialize data and model classes
        self.division_results = {}
        self.overall_accuracy = 0.0
        self.rmse = 0.0
        
        # Setup main containers
        self.setup_header()
        self.setup_content()
        
        # Initialize with the current year
        current_year = 2024  # Default to 2024 for demo purposes
        self.year_var.set(str(current_year))

        self.all_stats = DataFetcher.ALL_STATS
        self.selected_stats = list(DataFetcher.DEFAULT_STATS) # Make a mutable copy
        self.stat_vars = {}
        
    def setup_header(self):
        header_frame = ttk.Frame(self.root, style='Header.TFrame')
        header_frame.pack(fill=tk.X)
        
        # Logo and title
        title_label = ttk.Label(header_frame, text="MLB Win Prediction Visualizer", style='Header.TLabel')
        title_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Try to fetch and display MLB logo
        try:
            logo_url = "https://www.mlbstatic.com/team-logos/league-on-dark/1.svg"
            response = requests.get(logo_url)
            if response.status_code == 200:
                img_data = response.content
                img = Image.open(io.BytesIO(img_data))
                img = img.resize((80, 40), Image.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                logo_label = ttk.Label(header_frame, image=self.logo_img, background=self.header_color)
                logo_label.pack(side=tk.RIGHT, padx=10, pady=5)
        except:
            # If logo fetch fails, display a text alternative
            logo_alt = ttk.Label(header_frame, text="MLB", style='Header.TLabel')
            logo_alt.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def setup_content(self):
        # Main content area
        content_frame = ttk.Frame(self.root, padding=10)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(content_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Year selection
        year_frame = ttk.Frame(control_frame)
        year_frame.pack(fill=tk.X, pady=10)
        
        year_label = ttk.Label(year_frame, text="Projection Year:", style='Data.TLabel')
        year_label.pack(side=tk.LEFT, padx=5)
        
        self.year_var = tk.StringVar()
        years = [str(year) for year in range(2013, 2025) if year != 2020]
        year_combo = ttk.Combobox(year_frame, textvariable=self.year_var, values=years, width=6)
        year_combo.pack(side=tk.LEFT, padx=5)
        
        # Model selection
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=10)
        
        model_label = ttk.Label(model_frame, text="Model:", style='Data.TLabel')
        model_label.pack(side=tk.LEFT, padx=5)
        
        self.model_var = tk.StringVar(value="RidgeCV")
        model_values = [
            "LinearRegression", "RidgeCV", 
            "RandomForestRegressor", "GradientBoostingRegressor"
        ]
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=model_values, width=23)
        model_combo.pack(side=tk.LEFT, padx=5)
        
        # Run button
        run_button = ttk.Button(control_frame, text="Run Projection", command=self.run_projection, style='Accent.TButton')
        run_button.pack(fill=tk.X, pady=10)
        
        # Select features button
        features_button = ttk.Button(control_frame, text="Select Features", command=self.open_feature_selector)
        features_button.pack(fill=tk.X, pady=10)

        # Information display
        info_frame = ttk.LabelFrame(control_frame, text="Model Information", padding=10)
        info_frame.pack(fill=tk.X, pady=10)
        
        self.rmse_var = tk.StringVar(value="RMSE: -")
        rmse_label = ttk.Label(info_frame, textvariable=self.rmse_var, style='Data.TLabel')
        rmse_label.pack(anchor=tk.W, pady=2)
        
        self.accuracy_var = tk.StringVar(value="Overall Accuracy: -")
        accuracy_label = ttk.Label(info_frame, textvariable=self.accuracy_var, style='Data.TLabel')
        accuracy_label.pack(anchor=tk.W, pady=2)
        
        # Help
        help_button = ttk.Button(control_frame, text="Help/About", command=self.show_help)
        help_button.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Display Model Parameters
        self.param1_var = tk.StringVar(value="Parameter 1: -")
        self.param2_var = tk.StringVar(value="Parameter 2: -")

        param_frame = ttk.LabelFrame(control_frame, text="Model Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=10, expand=True)

        param1_label = ttk.Label(param_frame, textvariable=self.param1_var, style='Data.TLabel', wraplength=230)
        param1_label.pack(anchor=tk.W, pady=2)

        param2_label = ttk.Label(param_frame, textvariable=self.param2_var, style='Data.TLabel', wraplength=230)
        param2_label.pack(anchor=tk.W, pady=2)
        
        # Right area for results
        self.results_frame = ttk.Notebook(content_frame)
        self.results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Placeholder tabs
        self.tables_tab = ttk.Frame(self.results_frame, padding=10)
        self.graphs_tab = ttk.Frame(self.results_frame, padding=10)
        self.heatmap_tab = ttk.Frame(self.results_frame, padding=10)
        
        self.results_frame.add(self.tables_tab, text="Division Tables")
        self.results_frame.add(self.graphs_tab, text="Bar Charts")
        self.results_frame.add(self.heatmap_tab, text="Prediction Heatmap")
        
        # Status bar
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding=2)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Ready. Select a projection year and click 'Run Projection'")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X)
    
    def run_projection(self):
        # Clear previous results
        for widget in self.tables_tab.winfo_children():
            widget.destroy()
        for widget in self.graphs_tab.winfo_children():
            widget.destroy()
        for widget in self.heatmap_tab.winfo_children():
            widget.destroy()
            
        self.status_var.set("Running projection... Please wait.")
        self.root.update()
        
        try:
            projection_year = int(self.year_var.get())
            selected_model = self.model_var.get()
            
            # Run model projection (based on the original code)
            self.run_model(projection_year, selected_model)
            
            # Update status
            self.status_var.set(f"Projection completed for {projection_year} using {selected_model}")
            self.rmse_var.set(f"RMSE: {self.rmse:.2f}")
            self.accuracy_var.set(f"Overall Accuracy: {self.overall_accuracy*100:.2f}%")
            
            # Display results
            self.display_division_tables()
            self.display_bar_charts()
            self.display_heatmap()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred during projection.")
    
    def run_model(self, projection_year, selected_model):
        model_instance = self.create_model_instance(selected_model)

        main = Main(model=model_instance, selected_stats=self.selected_stats)
        results = main.run(projection_year)

        self.rmse = results["rmse"]
        self.division_accuracies = results["division_accuracies"]
        self.overall_accuracy = results["overall_accuracy"]
        self.results_df = results["results_df"]

        # update coefficients on gui
        model = results["model"]
        
        # Reset parameters before setting new ones
        self.param1_var.set("")
        self.param2_var.set("")

        if selected_model in ["LinearRegression", "RidgeCV"]:
            if hasattr(model, "intercept_") and hasattr(model, "coef_"):
                intercept_val = model.intercept_
                coef_val = model.coef_
                self.param1_var.set(f"Intercept: {intercept_val:.3f}")
                self.param2_var.set(f"Coefficients: {np.array_str(coef_val, precision=3, max_line_width=80)}")
        elif selected_model in ["RandomForestRegressor", "GradientBoostingRegressor"]:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_names = [f"{'p' if g == 'pitching' else 'h'}_{s}" for s, g in self.selected_stats]
                
                importances_dict = dict(zip(feature_names, importances))
                sorted_importances = sorted(importances_dict.items(), key=lambda item: item[1], reverse=True)
                
                importance_str = ", ".join([f"{name}: {val:.3f}" for name, val in sorted_importances])

                self.param1_var.set(f"n_estimators: {model.n_estimators}")
    
    def create_model_instance(self, model_name):
        if model_name == "LinearRegression":
            return LinearRegression()
        elif model_name == "RidgeCV":
            return RidgeCV()
        elif model_name == "RandomForestRegressor":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == "GradientBoostingRegressor":
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def display_division_tables(self):
        # Create a canvas with scrollbar for the tables
        canvas_frame = ttk.Frame(self.tables_tab)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Group by division
        division_groups = self.results_df.groupby('Division')
        
        # Division colors (for visual distinction)
        division_colors = {
            "AL East": "#BA0C2F",   # Red
            "AL Central": "#0C2340", # Navy
            "AL West": "#003831",    # Dark Green
            "NL East": "#002D72",    # Blue
            "NL Central": "#31006F", # Purple
            "NL West": "#FD5A1E"     # Orange
        }
        
        # Display each division in a table
        for i, (division, group) in enumerate(division_groups):
            # Division header
            division_frame = ttk.Frame(scrollable_frame, padding=10)
            division_frame.pack(fill=tk.X, pady=10)
            
            header_bg = division_colors.get(division, self.header_color)
            division_label = ttk.Label(
                division_frame, 
                text=f"{division} (Accuracy: {self.division_accuracies[division]*100:.2f}%)",
                background=header_bg,
                foreground="white",
                font=('Arial', 12, 'bold'),
                padding=5
            )
            division_label.pack(fill=tk.X)
            
            # Table for this division
            table_frame = ttk.Frame(division_frame)
            table_frame.pack(fill=tk.X, pady=5)
            
            # Table headers
            headers = ["Team", "Actual Wins", "Predicted Wins", "Difference"]
            for j, header in enumerate(headers):
                header_label = ttk.Label(
                    table_frame, 
                    text=header, 
                    font=('Arial', 11, 'bold'),
                    background="#DDDDDD",
                    padding=5,
                    borderwidth=1,
                    relief="solid"
                )
                header_label.grid(row=0, column=j, sticky="nsew")
                table_frame.grid_columnconfigure(j, weight=1)
            
            # Table data
            for k, (idx, row) in enumerate(group.iterrows(), 1):
                # Team name
                team_label = ttk.Label(
                    table_frame, 
                    text=idx, 
                    padding=5,
                    borderwidth=1,
                    relief="solid"
                )
                team_label.grid(row=k, column=0, sticky="nsew")
                
                # Actual wins
                actual_label = ttk.Label(
                    table_frame, 
                    text=f"{row['Actual Wins']:.0f}", 
                    padding=5,
                    borderwidth=1,
                    relief="solid"
                )
                actual_label.grid(row=k, column=1, sticky="nsew")
                
                # Predicted wins
                pred_label = ttk.Label(
                    table_frame, 
                    text=f"{row['Predicted Wins']:.1f}", 
                    padding=5,
                    borderwidth=1,
                    relief="solid"
                )
                pred_label.grid(row=k, column=2, sticky="nsew")
                
                # Difference with color coding
                diff = row['Difference']
                if diff > 0:
                    diff_color = "#E6FFE6"  # Light green (better than predicted)
                elif diff < 0:
                    diff_color = "#FFE6E6"  # Light red (worse than predicted)
                else:
                    diff_color = "#FFFFFF"  # White (exactly as predicted)
                    
                diff_label = ttk.Label(
                    table_frame, 
                    text=f"{diff:.1f}", 
                    padding=5,
                    borderwidth=1,
                    relief="solid",
                    background=diff_color
                )
                diff_label.grid(row=k, column=3, sticky="nsew")
    
    def display_bar_charts(self):
        # Create tabs for different divisions in the graphs tab
        graph_notebook = ttk.Notebook(self.graphs_tab)
        graph_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create a tab for each division
        division_groups = self.results_df.groupby('Division')
        
        for division, group in division_groups:
            # Create a frame for this division
            division_frame = ttk.Frame(graph_notebook)
            graph_notebook.add(division_frame, text=division)
            
            # Create the bar chart
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Sort teams by actual wins for better visualization
            group = group.sort_values('Actual Wins', ascending=False)
            
            # Set width and positions for bars
            width = 0.35
            ind = np.arange(len(group))
            
            # Create bars
            actual_bars = ax.bar(ind - width/2, group['Actual Wins'], width, label='Actual Wins', color='#1E88E5')
            predicted_bars = ax.bar(ind + width/2, group['Predicted Wins'], width, label='Predicted Wins', color='#FFC107')
            
            # Add labels and title
            ax.set_xlabel('Teams')
            ax.set_ylabel('Wins')
            ax.set_title(f'{division} - Actual vs Predicted Wins')
            ax.set_xticks(ind)
            ax.set_xticklabels(group.index, rotation=45, ha='right')
            ax.legend()
            
            # Add values above bars
            for bar in actual_bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
                
            for bar in predicted_bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            # Add accuracy information
            accuracy = self.division_accuracies[division]
            ax.annotate(f'Ranking Accuracy: {accuracy*100:.2f}%',
                        xy=(0.5, 0.02),
                        xycoords='figure fraction',
                        ha='center',
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
            
            fig.tight_layout()
            
            # Add the figure to the frame
            canvas = FigureCanvasTkAgg(fig, master=division_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_heatmap(self):
        # Create a heatmap showing the differences between actual and predicted wins
        fig = plt.figure(figsize=(12, 8))
        
        # Create a custom arrangement of subplots
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        division_order = [
            "AL East", "NL East",
            "AL Central", "NL Central",
            "AL West", "NL West"
        ]
        
        # Define a diverging color scheme (red to blue)
        cmap = plt.cm.RdBu_r  # Red for overperformance, Blue for underperformance
        norm = plt.Normalize(-20, 20)  # Adjust range as needed
        
        # Create subplots for each division
        for i, division in enumerate(division_order):
            ax = fig.add_subplot(gs[i//2, i%2])
            
            # Get the data for this division
            try:
                div_data = self.results_df[self.results_df['Division'] == division]
                
                # Sort by actual wins
                div_data = div_data.sort_values('Actual Wins', ascending=False)
                
                # Extract the data for plotting
                teams = div_data.index
                actual = div_data['Actual Wins']
                predicted = div_data['Predicted Wins']
                diff = div_data['Difference']
                
                # Create horizontal bar chart
                bars = ax.barh(teams, predicted, color=cmap(norm(diff)), alpha=0.7, label='Predicted')
                ax.barh(teams, actual, color='none', edgecolor='black', linestyle='--', 
                        linewidth=2, label='Actual')
                
                # Add labels for predicted wins
                for j, (team, pred, act) in enumerate(zip(teams, predicted, actual)):
                    # Determine the furthest point (max between predicted and actual)
                    furthest_point = max(pred, act)
                    
                    # Place the label after the furthest point
                    ax.text(furthest_point + 1, j, f'{pred:.1f}', 
                            va='center', ha='left', fontsize=8, color='black')
                
                # Set title and labels
                ax.set_title(f'{division} (Acc: {self.division_accuracies[division]*100:.1f}%)')
                ax.set_xlabel('Wins')
                ax.set_xlim(0, max(actual.max(), predicted.max()) * 1.1)
                
                # Add a grid for better readability
                ax.grid(axis='x', linestyle='--', alpha=0.6)
                
                # Legend (only for the first subplot)
                if i == 0:
                    ax.legend(loc='upper right', fontsize=8)
                    
            except Exception as e:
                # Handle any errors (like division not found)
                ax.text(0.5, 0.5, f"Error displaying {division}: {str(e)}", 
                    ha='center', va='center', transform=ax.transAxes)
        
        # Add a color bar to explain the color scale
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                        cax=cbar_ax)
        cbar.set_label('Actual - Predicted Wins\n(Red = Overperformed, Blue = Underperformed)')
        
        # Add overall title
        fig.suptitle(f'MLB Win Predictions - Year {self.year_var.get()}\nOverall Accuracy: {self.overall_accuracy*100:.2f}%, RMSE: {self.rmse:.2f}', 
                    fontsize=16)
        
        fig.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # Add the figure to the frame
        canvas = FigureCanvasTkAgg(fig, master=self.heatmap_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def open_feature_selector(self):
        # Create a new top-level window
        self.selector_window = tk.Toplevel(self.root)
        self.selector_window.title("Select Model Features")
        self.selector_window.geometry("400x500") # Set a default size

        # This main frame will hold the canvas and scrollbar
        container = ttk.Frame(self.selector_window)
        container.pack(fill="both", expand=True, padx=10, pady=5)

        # --- Create the Scrollable Area ---
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        # This frame will hold the checkboxes and be placed inside the canvas
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # --- Populate the Scrollable Frame with Checkboxes ---
        self.stat_vars = {}
        for stat_tuple in self.all_stats:
            stat_key = f"{stat_tuple[0]} ({stat_tuple[1]})"
            var = tk.BooleanVar()
            # Set the checkbox to checked if the stat is in the currently selected list
            var.set(stat_tuple in self.selected_stats)
            # Place the checkbox inside the scrollable_frame
            cb = ttk.Checkbutton(scrollable_frame, text=stat_key, variable=var)
            cb.pack(anchor="w", padx=10, pady=2)
            self.stat_vars[stat_key] = (var, stat_tuple)

        # --- Pack the canvas and scrollbar into the container ---
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- OK and Cancel buttons (placed outside the scrollable area) ---
        button_frame = ttk.Frame(self.selector_window, padding=(10, 5, 10, 10))
        button_frame.pack(fill="x")
        ok_button = ttk.Button(button_frame, text="OK", command=self.apply_feature_selection)
        ok_button.pack(side="right", padx=5)
        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.selector_window.destroy)
        cancel_button.pack(side="right")
    
    def apply_feature_selection(self):
        # Update the list of selected stats based on the user's choices
        new_selected_stats = []
        for stat_key, (var, stat_tuple) in self.stat_vars.items():
            if var.get():
                new_selected_stats.append(stat_tuple)
        self.selected_stats = new_selected_stats
        self.status_var.set(f"{len(self.selected_stats)} features selected.")
        self.selector_window.destroy()

    def show_help(self):
        help_text = """MLB Win Prediction Visualizer

    This application visualizes predictions from different regression models for MLB team wins.

    How to use:
    1. Select a projection year and model from the dropdowns.
    2. (Optional) Click "Select Features" to change the input stats.
    3. Click "Run Projection" to process the data and see the results.
    4. View results in the different tabs.

    Model information:
    - LinearRegression: Ordinary least squares linear regression.
    - RidgeCV: Ridge regression with built-in cross-validation.
    - RandomForestRegressor: An ensemble model using multiple decision trees.
    - GradientBoostingRegressor: An ensemble model building trees sequentially to correct prior errors.

    The accuracy is measured using Spearman rank correlation within divisions.
    RMSE (Root Mean Square Error) shows the overall prediction error in terms of wins.

    Data is sourced from the MLB StatsAPI.
    """
        messagebox.showinfo("Help & Information", help_text)
def main():
    root = tk.Tk()
    app = MLBPredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()