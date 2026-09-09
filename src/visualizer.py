import pandas as pd
import numpy as np
import os
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

logger = logging.getLogger(__name__)

class RiskVisualizer:
    """Generates institutional-grade HTML quantitative risk reports."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_risk_report(self, exposures_df: pd.DataFrame, risk_res: dict, filename: str = "risk_report.html") -> None:
        """
        Generates a comprehensive HTML tear sheet including KPI cards, 
        interactive Plotly charts, and a data table.
        """
        try:
            # 1. Build the interactive chart (Plotly)
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "bar"}, {"type": "domain"}]],
                subplot_titles=('Asset Factor Exposures (Beta)', 'Portfolio Risk Decomposition')
            )

            # Left panel: factor exposures
            for factor in ['beta_mkt', 'beta_smb', 'beta_hml']:
                if factor in exposures_df.columns:
                    fig.add_trace(
                        go.Bar(
                            x=exposures_df['symbol'],
                            y=exposures_df[factor],
                            name=factor.replace('beta_', '').upper()
                        ),
                        row=1, col=1
                    )

            # Right panel: risk decomposition
            factor_ratio = risk_res['factor_var_ratio']
            specific_ratio = 1.0 - factor_ratio

            fig.add_trace(
                go.Pie(
                    labels=['Systematic Factor Risk', 'Idiosyncratic Risk'],
                    values=[factor_ratio, specific_ratio],
                    hole=0.45,
                    marker=dict(colors=['#3b82f6', '#f97316'])  # modern blue/orange palette
                ),
                row=1, col=2
            )

            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                template='plotly_white',
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            
            # Convert the chart to an HTML <div> string
            plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

            # 2. Build the data-table HTML
            # Format the numeric values for display
            display_df = exposures_df.copy()
            for col in ['beta_mkt', 'beta_smb', 'beta_hml', 'r_squared']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            display_df['idiosyncratic_var'] = display_df['idiosyncratic_var'].apply(lambda x: f"{x:.6e}")

            table_html = display_df.to_html(classes="data-table", index=False)

            # 3. Assemble the full modern HTML report template
            sys_risk_pct = f"{factor_ratio * 100:.2f}%"
            idio_risk_pct = f"{specific_ratio * 100:.2f}%"
            total_var = f"{risk_res['total_variance']:.6e}"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            html_template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>AlphaRisk Engine - Risk Report</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
                <style>
                    body {{ font-family: 'Inter', sans-serif; background-color: #f1f5f9; color: #334155; margin: 0; padding: 40px 20px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }}
                    
                    .header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 20px; margin-bottom: 30px; }}
                    .header h1 {{ margin: 0; color: #0f172a; font-size: 28px; }}
                    .header p {{ margin: 5px 0 0 0; color: #64748b; font-size: 14px; }}
                    
                    .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 40px; }}
                    .kpi-card {{ background: #f8fafc; padding: 24px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center; }}
                    .kpi-label {{ font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
                    .kpi-value {{ font-size: 28px; font-weight: 700; color: #0f172a; margin-top: 10px; }}
                    
                    .chart-container {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 40px; background: #fff; }}
                    
                    .table-section h2 {{ font-size: 20px; color: #0f172a; margin-bottom: 15px; }}
                    .data-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
                    .data-table th, .data-table td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
                    .data-table th {{ background-color: #f8fafc; font-weight: 600; color: #475569; }}
                    .data-table tr:hover {{ background-color: #f1f5f9; }}
                    
                    .footer {{ margin-top: 50px; text-align: center; font-size: 13px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>AlphaRisk Engine - Quantitative Risk Report</h1>
                        <p>Multi-Factor Exposure & Portfolio Variance Decomposition</p>
                    </div>

                    <div class="kpi-grid">
                        <div class="kpi-card">
                            <div class="kpi-label">Portfolio Total Variance</div>
                            <div class="kpi-value">{total_var}</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-label">Systematic Risk (Factor Driven)</div>
                            <div class="kpi-value" style="color: #3b82f6;">{sys_risk_pct}</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-label">Idiosyncratic Risk (Asset Specific)</div>
                            <div class="kpi-value" style="color: #f97316;">{idio_risk_pct}</div>
                        </div>
                    </div>

                    <div class="chart-container">
                        {plot_html}
                    </div>

                    <div class="table-section">
                        <h2>Cross-Sectional Factor Exposures (Snapshot)</h2>
                        {table_html}
                    </div>

                    <div class="footer">
                        Generated by AlphaRisk Data Pipeline &bull; Computation Engine: NumPy / SQLite &bull; Report Time: {timestamp}
                    </div>
                </div>
            </body>
            </html>
            """

            # 4. Write the file
            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
                
            logger.info(f"Institutional HTML risk report generated: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")