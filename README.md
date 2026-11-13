# FreshStock AOR Simulation

An interactive Streamlit teaching demo for perishable-inventory replenishment.
It compares a paper-inspired dynamic-programming policy with every-period and
fixed-interval ordering using Poisson demand, FIFO inventory, expiry, and backlog.

> This is an educational surrogate, not an exact reproduction of the cited
> paper's Algorithm 1 or an optimal SDP solver. Its synthetic results do not
> establish the paper's reported optimality gap.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Deploy and share

1. Push this folder's files to a GitHub repository.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**.
3. Select the repository, its branch (usually `main`), and `streamlit_app.py`.
4. Deploy. Streamlit will provide a public `*.streamlit.app` URL to share.

The GitHub repository URL shares the source; the Streamlit URL opens the running
simulator.

## Custom demand input

Choose **Custom CSV upload** in the sidebar and upload `sample_demand.csv`.
The CSV must have `Period` and `Mean_Demand` columns, with consecutive periods
starting at 1.
