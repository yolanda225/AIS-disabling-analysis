"""
This script downloads gap and loitering events from the Global Fishing Watch API for two specified periods, 
along with vessel information for the vessels involved in the gap events. 
The data is saved to CSV files in the 'data/raw' directory.
"""
import asyncio
import ast

import pandas as pd
import gfwapiclient as gfw

from config import TOKEN, PERIODS

client = gfw.Client(access_token=TOKEN) # Global Fishing Watch API


async def download_gaps(start_date, end_date, outfile):
    result = await client.events.get_all_events(
        datasets=["public-global-gaps-events:latest"],
        start_date=start_date,
        end_date=end_date,
        vessel_types=["FISHING"],
    )
    df = result.df()
    df.to_csv(outfile, index=False)
    print(f"{start_date} - {end_date}: {len(df)} gap events -> {outfile}")
    return df


async def download_vessel_info(df_gaps, outfile):
    # Get all unique vessel IDs from the gap events
    df_gaps['vessel'] = df_gaps['vessel'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    vessel_ids = df_gaps['vessel'].apply(lambda x: x['id']).unique().tolist()
    print(f"Unique vessels to look up: {len(vessel_ids)}")

    # Look up gear type for each vessel
    all_vessels = []
    chunk_size = 20
    for i in range(0, len(vessel_ids), chunk_size):
        chunk = vessel_ids[i:i + chunk_size]
        try:
            result = await client.vessels.get_vessels_by_ids(ids=chunk)
            all_vessels.append(result.df())
        except Exception as e:
            print(f"Error on chunk {i}: {e}")
    vessel_info = pd.concat(all_vessels, ignore_index=True)

    # explode the nested list into rows, normalize to flat columns
    df_combined = vessel_info['combined_sources_info'].explode()
    df_combined = pd.json_normalize(df_combined)
    df_combined.to_csv(outfile, index=False)
    print(f"{len(df_combined)} vessel records -> {outfile}")


async def download_loitering(start_date, end_date, outfile):
    result = await client.events.get_all_events(
        datasets=["public-global-loitering-events:latest"],
        start_date=start_date,
        end_date=end_date,
        vessel_types=["CARRIER"],
    )
    df = result.df()
    df.to_csv(outfile, index=False)
    print(f"{start_date} - {end_date}: {len(df)} loitering events -> {outfile}")


async def main():
    # Period A
    period_a = "2017_2019"
    start_a, end_a = PERIODS[period_a]
    df_a = await download_gaps(start_a, end_a, f"data/raw/gaps_{period_a}.csv")
    await download_vessel_info(df_a, f"data/raw/vessel_info_{period_a}.csv")

    # Period B
    period_b = "2023_2025"
    start_b, end_b = PERIODS[period_b]
    df_b = await download_gaps(start_b, end_b, f"data/raw/gaps_{period_b}.csv")
    await download_vessel_info(df_b, f"data/raw/vessel_info_{period_b}.csv")

    # Loitering events
    await download_loitering(start_a, end_a, f"data/raw/loitering_{period_a}.csv")
    await download_loitering(start_b, end_b, f"data/raw/loitering_{period_b}.csv")


if __name__ == "__main__":
    asyncio.run(main())