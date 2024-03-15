from xspec import AllModels, AllData, Model, Plot, FakeitSettings

fs1 = FakeitSettings(response="files/ni5050300117mpu7.rmf", arf="files/ni5050300117mpu7.arf", exposure="1e5", fileName='test.fak')

# Clear existing XSPEC models and data
AllModels.clear()
AllData.clear()

# Create the model
model_name = "TBabs*(rdblur*rfxconv*comptb + diskbb + comptb)"
model = Model(model_name)
AllModels.setPars(model)
AllData.fakeit(1, fs1)
AllData.ignore('**-0.3')
AllData.ignore('10.-**')
# Set up the energy range of interest for plotting
Plot.device = "/xs"
Plot.xAxis = "keV"
Plot.show()
Plot('data')
energy = Plot.x()
energy_e = Plot.xErr()
flux = Plot.y()
flux_e = Plot.yErr()