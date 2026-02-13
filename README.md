# Compute Specs DB

A public, searchable catalog of HPC and datacenter compute specifications with a web UI and REST API.

Live site: https://computespecsdb.com/

## What this repo provides
- Searchable specs table for server-class CPUs (GPU and ARM data planned)
- Visualizations and summary stats
- Public API for programmatic access

## Who this is for
- HPC and datacenter practitioners
- Researchers and analysts
- Anyone comparing server-class compute hardware

## Website
Browse the public site for data, visualizations, and API links:
https://computespecsdb.com/

## API access
The API supports:
- Listing CPUs
- Searching by model, family, or codename
- Accessing summary stats
- Exporting data (CSV/Excel)

See the API docs link from the website.

## Data scope
This project currently focuses on HPC and datacenter CPUs (Intel Xeon, AMD EPYC, Opteron, etc.). GPU and ARM processor data is planned for future releases. If you spot missing entries or incorrect specs, contributions are welcome.

## Data validation
All specifications have been manually validated by [me](@elokwentnie) against official manufacturer sources and trusted hardware databases:

- [Intel — Official product specifications](https://www.intel.com/content/www/us/en/products/overview.html)
- [AMD — Official processor specifications](https://www.amd.com/en/products/specifications/processors.html)
- [TechPowerUp — CPU Database](https://www.techpowerup.com/)
- [PassMark — CPU Benchmark Charts](https://www.cpubenchmark.net/)

Of course, it's possible I made a mistake, so if you find any inaccuracies, please [open an issue](https://github.com/elokwentnie/compute-specs-db/issues) on GitHub.

## Project structure
```
compute-specs-db/
├── app.py                      # FastAPI application
├── auth.py                     # JWT authentication helpers
├── database.py                 # SQLAlchemy models/config
├── import_data.py              # CSV import utility
├── utils.py                    # Helpers (generation/codename logic)
├── requirements.txt            # Python dependencies
├── Procfile                    # Render deployment entrypoint
├── cpu_spec_validated.csv      # Source data file
├── .github/
│   └── ISSUE_TEMPLATE/
│       ├── new-cpu-request.yml # Template for requesting new CPU entries
│       └── report-data-error.yml # Template for reporting data errors
└── static/
    ├── index.html              # Public web interface
    ├── visualizations.html     # Charts and insights
    ├── admin.html              # Admin UI (protected)
    ├── css/
    │   ├── common.css          # Shared styles
    │   ├── index.css           # Data table styles
    │   ├── admin.css           # Admin panel styles
    │   └── visualizations.css  # Charts page styles
    └── images/
        └── server-logo.png     # Site favicon / logo
```

## Getting started

### Prerequisites
- Python 3.11+

### Local development

```bash
# Clone the repository
git clone https://github.com/elokwentnie/compute-specs-db.git
cd compute-specs-db

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy the example environment file and adjust as needed
cp .env.example .env

# Run the development server
uvicorn app:app --reload
```

The app will be available at `http://localhost:8000`. The database is auto-populated from `cpu_spec_validated.csv` on first run.

## Contributing

Contributions are encouraged, especially:
- Adding missing CPUs (GPU and ARM data also welcome)
- Fixing incorrect specs
- Improving data quality or coverage
- UI/UX improvements

### Request a new CPU entry
Use the **[New CPU Request](https://github.com/elokwentnie/compute-specs-db/issues/new?template=new-cpu-request.yml)** issue template. Fill in as many fields as you can and include a link to the official spec page or a trusted hardware database.

### Report a data error
Use the **[Report Data Error](https://github.com/elokwentnie/compute-specs-db/issues/new?template=report-data-error.yml)** issue template. Specify which field is wrong, the current and correct values, and a source URL.

### Code contributions
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a clear description of your changes.
4. Include sources or references for any new data.

## Git history

The commit history was squashed into a single initial commit before making this repository public. The original development history contained iterative changes, debugging steps, and configuration experiments that were not meaningful for external contributors. This gives the repo a clean starting point while preserving the final, reviewed state of all code and data.

## Security

To report a security vulnerability, please see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the [MIT License](LICENSE).
