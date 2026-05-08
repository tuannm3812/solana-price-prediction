import streamlit as st

from views.solana import render_solana_tab

st.set_page_config(
    page_title="Crypto Investment Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


def main() -> None:
    st.title("Cryptocurrency Investment Dashboard")
    st.markdown("Explore market data and machine-learning predictions for supported assets.")

    tab_sol, tab_btc, tab_eth, tab_xrp = st.tabs(
        ["Solana (SOL)", "Bitcoin (BTC)", "Ethereum (ETH)", "Ripple (XRP)"]
    )

    with tab_sol:
        render_solana_tab()

    with tab_btc:
        st.info("Bitcoin module is not configured in this build.")

    with tab_eth:
        st.info("Ethereum module is not configured in this build.")

    with tab_xrp:
        st.info("Ripple module is not configured in this build.")

    st.divider()
    st.caption("Solana next-day high prediction project")


if __name__ == "__main__":
    main()
