from src.base_model.simson_base_model import load_simson_base_model, ENV_PID, BOF_PID, EAF_PID, FORM_PID, IP_PID, \
    FABR_PID, USE_PID, BUF_PID, OBS_PID, EOL_PID, RECYCLE_PID, SCRAP_PID, EXC_PID, FBUF_PID
from src.read_data.load_data import load_region_names_list
from matplotlib import pyplot as plt
import numpy as np

model = load_simson_base_model()
scrap_bof = model.get_flowV(SCRAP_PID, BOF_PID)
scrap_eaf = model.get_flowV(SCRAP_PID, EAF_PID)
iron_production = model.get_flowV(ENV_PID, BOF_PID)

scrap_in_production = scrap_eaf + scrap_bof
production = scrap_in_production + iron_production
scrap_share_production = np.divide(scrap_in_production,
                                   production,
                                   out=np.zeros_like(scrap_in_production),
                                   where=production != 0)
regions = load_region_names_list()
years = range(1900, 2101)

for r, region in enumerate(regions):
    plt.plot(years,
             scrap_share_production[:, 0, r, 1])  # choose all years, 'Fe'/Iron production, region r and scenario SSP2

plt.xlabel('Years')
plt.ylabel('Scrap share (%)')
plt.legend(regions)
plt.title('Scrap share in production over regions')
plt.show()
