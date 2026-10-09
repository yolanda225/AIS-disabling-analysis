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
    """Download AIS gap events for a date range and save them to CSV.

    Args:
        start_date (str): Start of the range ("YYYY-MM-DD").
        end_date (str): End of the range ("YYYY-MM-DD").
        outfile (str): CSV path to write the gap events to.

    Returns:
        pandas.DataFrame: The downloaded gap events.
    """
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
    """Look up gear type and other vessel info for every vessel in a gap-events
    DataFrame, and save the records to CSV.

    Args:
        df_gaps (pandas.DataFrame): Gap events with a 'vessel' column.
        outfile (str): CSV path to write the vessel records to.

    Returns:
        None: Writes outfile.
    """
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
    """Download carrier-vessel loitering events for a date range and save to CSV.

    Args:
        start_date (str): Start of the range ("YYYY-MM-DD").
        end_date (str): End of the range ("YYYY-MM-DD").
        outfile (str): CSV path to write the loitering events to.

    Returns:
        None: Writes outfile.
    """
    result = await client.events.get_all_events(
        datasets=["public-global-loitering-events:latest"],
        start_date=start_date,
        end_date=end_date,
        vessel_types=["CARRIER"],
    )
    df = result.df()
    df.to_csv(outfile, index=False)
    print(f"{start_date} - {end_date}: {len(df)} loitering events -> {outfile}")


def add_gear_type(gaps_df, vessel_df, outfile):
    """Merge each gap event's gear type in from vessel info, and save the result to CSV.

    Args:
        gaps_df (pandas.DataFrame): Gap events with a 'vessel' column.
        vessel_df (pandas.DataFrame): Vessel records with 'vessel_id' and 'gear_types'.
        outfile (str): CSV path to write the merged events to.

    Returns:
        None: Writes outfile.
    """
    vessel_df = vessel_df.copy()
    vessel_df['gear_type'] = vessel_df['gear_types'].apply(
        lambda g: (g if isinstance(g, list) else ast.literal_eval(g))[0]['name']
        if len(g if isinstance(g, list) else ast.literal_eval(g)) > 0 else None
    )

    gaps_df = gaps_df.copy()
    gaps_df['vessel'] = gaps_df['vessel'].apply(lambda v: v if isinstance(v, dict) else ast.literal_eval(v))
    gaps_df['vessel_id'] = gaps_df['vessel'].apply(lambda v: v['id'])

    gaps_df = gaps_df.merge(
        vessel_df[['vessel_id', 'gear_type']], on='vessel_id', how='left'
    )

    gaps_df.to_csv(outfile, index=False)
    print(f"{len(gaps_df)} events with gear type -> {outfile}")


GLOBAL_GEOJSON = {
    "type": "Polygon",
    "coordinates": [[
        [-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]
    ]]
}


async def download_fishing_effort_gridded(start_date, end_date, outfile, spatial_resolution="LOW"):
    """Download gridded fishing-effort hours one year at a time grouped by geartype
    and save the combined result to CSV.

    Args:
        start_date (str): Start of the range ("YYYY-MM-DD").
        end_date (str): End of the range ("YYYY-MM-DD").
        outfile (str): CSV path to write the gridded fishing effort to.
        spatial_resolution (str, optional): GFW grid resolution. Defaults to "LOW".

    Returns:
        pandas.DataFrame: The combined gridded fishing effort across all years.
    """
    years = pd.date_range(start_date, end_date, freq="YS")
    running = None

    for year_start in years:
        year_end = min(pd.Timestamp(end_date), year_start + pd.DateOffset(years=1) - pd.DateOffset(days=1))
        result = await client.fourwings.create_report(
            datasets=["public-global-fishing-effort:latest"],
            spatial_resolution=spatial_resolution,
            temporal_resolution="ENTIRE",
            group_by="FLAGANDGEARTYPE",
            spatial_aggregation=False,
            start_date=year_start.strftime("%Y-%m-%d"),
            end_date=year_end.strftime("%Y-%m-%d"),
            geojson=GLOBAL_GEOJSON,
        )
        df_year = result.df()[['lat', 'lon', 'flag', 'gear_type', 'hours']]  # keep only needed columns
        df_year['lat'] = df_year['lat'].astype('float32')
        df_year['lon'] = df_year['lon'].astype('float32')
        df_year['hours'] = df_year['hours'].astype('float32')
        print(f"{year_start.date()} - {year_end.date()}: {len(df_year)} rows")

        if running is None:
            running = df_year.groupby(['lat', 'lon', 'flag', 'gear_type'], as_index=False, dropna=False)['hours'].sum()
        else:
            running = pd.concat([running, df_year], ignore_index=True)
            running = running.groupby(['lat', 'lon', 'flag', 'gear_type'], as_index=False, dropna=False)['hours'].sum()

        del df_year, result  # free memory
        import gc; gc.collect()

    running.to_csv(outfile, index=False)
    print(f"Total: {len(running)} gridded rows -> {outfile}")
    return running


async def main():
    # Period A
    period_a = "2017_2019"
    start_a, end_a = PERIODS[period_a]
    # df_a = await download_gaps(start_a, end_a, f"data/raw/gaps_{period_a}.csv")
    # await download_vessel_info(df_a, f"data/raw/vessel_info_{period_a}.csv")

    # Period B
    period_b = "2023_2025"
    start_b, end_b = PERIODS[period_b]
    # df_b = await download_gaps(start_b, end_b, f"data/raw/gaps_{period_b}.csv")
    # await download_vessel_info(df_b, f"data/raw/vessel_info_{period_b}.csv")

    # # Loitering events
    # await download_loitering(start_a, end_a, f"data/raw/loitering_{period_a}.csv")
    # await download_loitering(start_b, end_b, f"data/raw/loitering_{period_b}.csv")

    # # Add gear type to gaps
    # df_a = pd.read_csv(f"data/raw/gaps_{period_a}.csv")
    # vessel_df_a = pd.read_csv(f"data/raw/vessel_info_{period_a}.csv")
    # df_a = add_gear_type(df_a, vessel_df_a, "data/processed/gaps_2017_2019_filtered.csv")

    # df_b = pd.read_csv(f"data/raw/gaps_{period_b}.csv")
    # vessel_df_b = pd.read_csv(f"data/raw/vessel_info_{period_b}.csv")
    # df_b = add_gear_type(df_b, vessel_df_b, "data/processed/gaps_2023_2025_filtered.csv")

    df_fishing_gridded_a = await download_fishing_effort_gridded(start_a, end_a, f"data/raw/fishing_gridded_{period_a}.csv")
    df_fishing_gridded_b = await download_fishing_effort_gridded(start_b, end_b, f"data/raw/fishing_gridded_{period_b}.csv")


if __name__ == "__main__":
    asyncio.run(main())