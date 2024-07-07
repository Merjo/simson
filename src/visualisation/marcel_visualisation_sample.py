from src.base_model.simson_base_model import load_simson_base_model, ENV_PID, BOF_PID, EAF_PID, FORM_PID, IP_PID, \
    FABR_PID, USE_PID, BUF_PID, OBS_PID, EOL_PID, RECYCLE_PID, SCRAP_PID, EXC_PID, FBUF_PID
from src.read_data.load_data import load_region_names_list
from matplotlib import pyplot as plt
import numpy as np
from src.tools.config import cfg


model = load_simson_base_model()
scrap_bof = model.get_flowV(SCRAP_PID, BOF_PID)
scrap_eaf = model.get_flowV(SCRAP_PID, EAF_PID)
iron_production = model.get_flowV(ENV_PID, BOF_PID)

in_use_good_outflow = model.get_flowV(USE_PID, BUF_PID)

regions = load_region_names_list()
years = range(1900, 2101)

# Define the goods categories
goods = ['Construction','Machinery','Products','Transport']

#scrap share in production
scrap_in_production = scrap_eaf + scrap_bof
scrap_in_production_world = scrap_in_production.sum(axis=2)
production = scrap_in_production + iron_production
production_world = scrap_in_production_world + iron_production.sum(axis=2)
scrap_share_production = np.divide(scrap_in_production,
                                   production,
                                   out=np.zeros_like(scrap_in_production),
                                   where=production != 0)
scrap_share_production_world = np.divide(scrap_in_production_world,
                                   production_world,
                                   out=np.zeros_like(scrap_in_production_world),
                                   where=production_world != 0)

print(scrap_share_production.shape)
print(scrap_share_production_world.shape)

for r, region in enumerate(regions):
    plt.plot(years,
             scrap_share_production[:, 0, r, 1])  # choose all years, 'Fe'/Iron production, region r and scenario SSP2

plt.xlabel('Years')
plt.ylabel('Scrap share (%)')
plt.legend(regions)
plt.title('Scrap share in production over regions')
plt.show()

#global scrap share in production
for i in range (0,201):
    plt.plot(years, scrap_share_production_world[:, 0, 1])
region='World'
plt.xlabel('Years')
plt.ylabel('Scrap Share in Production (%)')
plt.legend(region)
plt.title(f"Scrap share in production worldwide {cfg.recycling_strategy}-model")
plt.show()

#EoL outflow world

in_use_good_outflow_world = in_use_good_outflow.sum(axis=2)

# Extract the relevant data for iron (element 0) and scenario 1
data = in_use_good_outflow_world[:, 0, :, 1]

#old sorting
'''outflow_transport = in_use_good_outflow_world[108:109, 0, 0:1, 1]
outflow_machinery = in_use_good_outflow_world[108:109, 0, 1:2, 1]
outflow_construction = in_use_good_outflow_world[108:109, 0, 2:3, 1]
outflow_products = in_use_good_outflow_world[108:109, 0, 3:4, 1]'''
#new alphabetical sorting
outflow_construction = in_use_good_outflow_world[108:109, 0, 0:1, 1]
outflow_machinery = in_use_good_outflow_world[108:109, 0, 1:2, 1]
outflow_products = in_use_good_outflow_world[108:109, 0, 2:3, 1]
outflow_transport = in_use_good_outflow_world[108:109, 0, 3:4, 1]
check_sum_outflow = outflow_transport + outflow_machinery + outflow_construction + outflow_products
print('outflow_transport 2008: ', outflow_transport)
print('outflow_machinery 2008: ', outflow_machinery)
print('outflow_construction 2008: ', outflow_construction)
print('outflow_products 2008: ', outflow_products)
#print('check_sum_outflow: ', check_sum_outflow)

recov_t = outflow_transport *0.9
recov_m = outflow_machinery *0.9
recov_c = outflow_construction *0.85
recov_p = outflow_products *0.5

copper_transport = recov_t * 0.003
copper_machinery = recov_m * 0.0025
copper_construction = recov_c * 0.001
copper_products = recov_p * 0.004
sum_copper = copper_transport + copper_machinery + copper_construction + copper_products
print('sum_copper: ', sum_copper) # 0.7Mt is the result of Daehn et al. (2017)

# derzeit sind nachfolgende Kategorien vertauscht
# transport= construction, machinery = machinery, construction = products, products = transport

sum_recov = recov_p + recov_c +recov_m +recov_t
recov_rate = sum_recov /check_sum_outflow
print('recov_rate: ', recov_rate)

#eigentliche copper rate


# Create a stacked area plot
fig, ax = plt.subplots()

# Create the stack plot
ax.stackplot(years, data.T, labels=goods, colors=['black', 'grey', 'brown', 'orange'])

# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Outflow (t)')
ax.set_title('In-Use Good Outflows (Iron, SSP2)')
ax.legend(loc='upper left')

# Calculate and plot values_2008
values_2008 = np.sum(data[108]) / 1000000  # Assuming 108 corresponds to the year 2008
print(f"Total outflow in 2008: {values_2008:.2f}Mt")
plt.axvline(x=2008, linestyle='--', color='black')
plt.text(2008, plt.ylim()[1], f'{values_2008:.2f}M', color='black', verticalalignment='top')

# Show the plot
plt.show()