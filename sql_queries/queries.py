def token_prices(today):

  beginning = f"'{today}'"
  print('beginning', beginning)
  
  prices_query =f"""

    WITH addresses AS (
        SELECT column1 AS token_address 
        FROM (VALUES
            (LOWER('0x2260fac5e5542a773aa44fbcfedf7c193bc2c599')), --wbtc
            (LOWER('0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2')), --weth
            (LOWER('0x45804880de22913dafe09f4980848ece6ecbaf78')), --paxg
            (LOWER('0xd46ba6d942050d489dbd938a2c909a5d5039a161')) --ampl

        ) AS tokens(column1)
    )

    select hour,
        symbol,
        price
    from ethereum.price.ez_prices_hourly
    where token_address in (select token_address from addresses)
    and hour >= date('2019-05-01')
    order by hour desc, symbol 


    """
  return prices_query

def volume(today):

    beginning = f"'{today}'"
    print('beginning', beginning)

    volume_query=f""""

    select
    dt,
    sum(volume) as volume
    from
    (
        select
        date_trunc('day', block_timestamp) as dt,
        sum(AMOUNT_IN_USD) as volume
        from
        ethereum.defi.ez_dex_swaps
        where
        AMOUNT_IN_USD is not null and date_trunc('day', block_timestamp) >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
        group by
        1
        union
        all
        select
        date_trunc('day', block_timestamp) as dt,
        sum(AMOUNT_IN_USD) as volume
        from
        arbitrum.defi.ez_dex_swaps
        where
        AMOUNT_IN_USD is not null and date_trunc('day', block_timestamp) >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
        group by
        1
        union
        all
        select
        date_trunc('day', block_timestamp) as dt,
        sum(AMOUNT_IN_USD) as volume
        from
        optimism.defi.ez_dex_swaps
        where
        AMOUNT_IN_USD is not null and date_trunc('day', block_timestamp) >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
        group by
        1
        union
        all
        select
        date_trunc('day', block_timestamp) as dt,
        sum(AMOUNT_IN_USD) as volume
        from
        base.defi.ez_dex_swaps
        where
        AMOUNT_IN_USD is not null and date_trunc('day', block_timestamp) >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
        group by
        1
        union
        all
        select
        date_trunc('day', block_timestamp) as dt,
        sum(AMOUNT_IN_USD) as volume
        from
        polygon.defi.ez_dex_swaps
        where
        AMOUNT_IN_USD is not null and date_trunc('day', block_timestamp) >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
        group by
        1
        union
        all
        select
        date_trunc('day', block_timestamp) as dt,
        sum(SWAP_FROM_AMOUNT_USD) as volume
        from
        solana.defi.ez_dex_swaps
        where
        SWAP_FROM_AMOUNT_USD is not null and date_trunc('day', block_timestamp) >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
        group by
        1
        union
        all
        select
        date_trunc('day', block_timestamp) as dt,
        sum(AMOUNT_IN_USD) as volume
        from
        avalanche.defi.ez_dex_swaps
        where
        AMOUNT_IN_USD is not null and date_trunc('day', block_timestamp) >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
        group by
        1
    )
    where
    dt >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
    group by
    1
    order by
    dt desc
    
        """
    return volume_query


   

