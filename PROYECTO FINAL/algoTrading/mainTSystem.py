import os 


def mensaje_titulo(text):
    print("="*len(text))
    print(text.upper())
    print("="*len(text))

def menu():
    print("""Elija 1 opcion:
    1 barsComputation
    2 featComputation
    3 backtestComputation
    4 Salir
    """)

def solicitar_opcion():
    menu()
    opcion = input("Opcion: ").strip()
    while opcion not in ['1', '2', '3', '4']:
        print("Ingrese una opcion valida.")
        print("")
        menu()
        opcion = input("Opcion: ").strip()
    return opcion

mensaje_titulo("Bienvenido a Trading Algoritmico - NFLX US Equity")
opcion = solicitar_opcion()


while opcion != '4':
    if opcion=='1':
        if all(file in os.listdir('./tickdata') for file in ['NFLX US Equity.zarr']):
            os.system('python barsComputation.py')
        else:
            print("El archivo 'NFLX US Equity.zarr' no se encuentra en tickdata")
            
    if opcion=='2':
        if all(file in os.listdir('./bars') for file in ['NFLX US Equity.csv']):
            os.system('python featComputation.py')
        else:
            print("Falta ejecutar 'barsComputation'")
    
    if opcion=='3':
        if all(file in os.listdir('./ProcessedFeatures') for file in ['aligned_labels.csv', 'aligned_tprices.csv', 'NFLX US Equity_Features_Global.csv', 'NFLX US Equity_Features_Selected.csv', 'scaler_pickle.pkl', 'stacked_scaled_features.csv']):
            os.system('python backtestComputation.py')
        else:
            print("Falta ejecutar 'featComputation'")
            
    print("")
    opcion = solicitar_opcion()

mensaje_titulo("Fin, hasta pronto.\nCredito: Aaron Calderon Guillermo")


