import netCDF4
ds = netCDF4.Dataset('fort.63.nc')
print('fort.63.nc vars:', list(ds.variables.keys()))
print('Time units:', ds.variables['time'].units if 'time' in ds.variables else 'No time variable')
print('Time shape:', ds.variables['time'].shape if 'time' in ds.variables else 'No time variable')
ds2 = netCDF4.Dataset('fort.74.nc')
print('fort.74.nc vars:', list(ds2.variables.keys()))
