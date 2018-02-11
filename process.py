import json
import os.path
import numpy
import asyncio
from aiohttp import ClientSession
import gzip
from collections import namedtuple

with open('indicators.txt', 'r') as f:
    raw_indicators = f.read()

indicators = raw_indicators.strip().split("\n")

"""
indicators = [
    'SP.DYN.CDRT.IN', # death rate
    'SP.DYN.CBRT.IN', # birth rate
    'FR.INR.RINR', # real interest rate
    'SP.POP.TOTL', # population
    'NY.GDP.PCAP.KD', # GDP per capita (constant 2010 US$)
    'FP.CPI.TOTL', # consumer price index
]
"""

indicatorToI = {}
for i, indicator in enumerate(indicators):
    indicatorToI[indicator] = i

def year_to_time(year):
    return int(year)

async def get(url):
    async with ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

async def fetch_indicator(indicator):
    fname = 'tmp/{}.json.gz'.format(indicator)

    if os.path.isfile(fname):
        print('Reading indicator from disk cache: {}'.format(indicator))
        with gzip.open(fname, 'rb') as f:
            return json.load(f)

    print('Fetching indicator: {}'.format(indicator))
    url = 'https://api.worldbank.org/v2/countries/all/indicators/{}?format=json&per_page=20000'.format(indicator)
    resp = await get(url)
    raw = json.loads(resp)
    meta = raw[0]
    if meta['pages'] > 1:
        print('got more than 1 page', meta)
        exit()

    indicators = raw[1]
    filtered = [
        i
        for i in indicators
        if len(i.get('countryiso3code')) > 0 and i.get('value') is not None
    ]
    for i in filtered:
        i['time'] = year_to_time(i['date'])


    with gzip.open(fname, 'wb') as f:
        json.dump(filtered, f)

    return filtered

countries = {}

loaded = 0

Datum = namedtuple('Datum', ['indicator_id', 'value', 'time'])

async def load_indicator(indicator_id, indicator):
    global countries, loaded, indicators

    data = await fetch_indicator(indicator)

    loaded += 1
    print('Loaded {}/{}: {}'.format(loaded, len(indicators), indicator))

    for i in data:
        country_code = i['countryiso3code']
        country_data = countries.get(country_code, [])
        country_data.append(
            Datum(indicator_id=indicator_id, value=i['value'], time=i['time']))
        countries[country_code] = country_data

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

async def load_all_indicators():
    await asyncio.wait([load_indicator(i, indicator) for i, indicator in enumerate(indicators)])

loop = asyncio.get_event_loop()
loop.run_until_complete(load_all_indicators())

print('Loaded all indicators, processing now...')

for k, v in countries.items():
    v = sorted(v, key=lambda i: i.time)
    countries[k] = v

trainX = numpy.zeros([0, len(indicators)*2], numpy.float32)
trainy = []

def add_example(last, current_time, changed):
    global trainX, trainy, indicators

    xi = []
    for datum in last:
        xi.append(datum.value)
        xi.append(current_time - datum.time)

    trainy.append(changed)
    trainX = numpy.vstack([trainX, xi])

countries_done = 0

for country, data in countries.items():
    last = {}
    last_time = 0

    countries_done += 1
    print('Processing {}/{}: {}'.format(countries_done, len(countries), country))

    current = [
        Datum(indicator_id=i, value=0, time=0)
        for i, indicator in enumerate(indicators)
    ]

    changed = {}

    current_time = 0

    for datum in data:
        if datum.time != current_time:
            if last_time > 0:
                add_example(last, current_time, changed)

            last_time = current_time
            last = current[:]
            current_time = datum.time
            changed = {}

        indicator_id = datum.indicator_id
        current[indicator_id] = datum
        changed[indicator_id] = datum.value

print('Num training examples = {}'.format(len(trainy)))

with gzip.open('tmp/data.json.gz', "wt", encoding="utf8") as f:
    json.dump({
        'indicators': indicators,
        'trainX': trainX.tolist(),
        'trainy': trainy
    }, f)
