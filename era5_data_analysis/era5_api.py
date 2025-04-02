import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': 'Mean sea level pressure',
        'year': '2023',
        'month': '11',
        'day': [
            '01'
        ],
        'time': [
            '00:00'
        ],
        'area': [
            49.5, 16, 47.5,
            23,
        ],
        'format': 'netcdf',
    },
    'download2.nc')
