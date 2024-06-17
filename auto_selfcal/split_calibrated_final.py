from casatools import msmetadata as msmdtool

msmd = msmdtool()

def split_calibrated_final():
    msmd.open("calibrated_final.ms")
    for i in range(msmd.nobservations()):
        split("calibrated_final.ms", outputvis=msmd.schedule(i)[1].split(" ")[1].replace("/","_").replace(":","_")+"_targets.ms", \
                observation=i, intent="*OBSERVE_TARGET*", \
                spw=','.join(msmd.spwsforscan(msmd.scansforintent("*OBSERVE_TARGET*", obsid=i)[0], obsid=i).astype(str)), \
                antenna=','.join(msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*", obsid=i)[0], obsid=i).astype(str)), \
                datacolumn="data")
    msmd.close()
