Tutorial veloce PEAN:
1) Attivare python virtual environment andando su lmautino/PeanImp/ ed eseguire comando
        source venv/bin/activate
2) Se si vuole creare un dataset da una cartella di file .pcap, oltre a quelle già installate, scaricare i .pcap dentro a Implementation/pcapD   atasets. (Tutto il codice che usiamo sta dentro Implementation, eccetto venv ciò che sta fuori può essere eliminato.)
   Scaricato il .pcap si può creare un dataset o per il pretraining o per il modello principale (i dataset hanno formati diversi per questo bi   sogna crearne due tipi)
    Per il pretrain eseguire

        python preTrainQuick.py --pcap_folder='./pcapDatasets/[cartella pcap]' --new=True

    **Se non si vuole addestrare un nuovo modello** togliere l'argomento --new=True, in tal caso il codice automaticamente va a prendere l'ult
    -imo modello addestrato. 

3) Per il modello generale stessa cosa solo che si esegue il comando

        python mainQuick.py 
        
    e come per il pretrain aggiungere il parametro --new=True per addestrare un nuovo modello altrimenti viene caricato quello pre esistente.
    Per caricare un nuovo dataset si usa sempre --pcap_folder='./pcapDatasets/[cartella pcap]'. 
    Il modello più grande in automatico carica il modello di pretrain. 


4) Oltre alle metriche riportate su linea di comando, pretrain salva in ./Implementation/runs i log del training e della evaluation mentre il    modello main salva su Implementation/Model/log.
   Questi log vanno scaricati in locale e aperti lanciando tensorboard --logdir=[cartella logs] e aprendo su un browser all'indirizzo locale
   che tensorboard provvvede (di solito localhost:6006)
