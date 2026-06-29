REQUIRED = [

    "AvgDiffElectronFlux",

    "AvgDiffProtonFlux",

    "AvgIntElectronFlux"

]

def validate(ds):

    missing=[]

    for var in REQUIRED:

        if var not in ds.variables:

            missing.append(var)

    return missing