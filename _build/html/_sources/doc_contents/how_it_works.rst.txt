Fasi del modello
================
Qui vengono spiegate nel dettaglio le diverse fasi del modello


Pretrain
~~~~~~~~
Nel pretrain essenzialmente alleniamo il modello a creare associazioni tra i byte per poi concentrare queste 'informazioni' all'interno di un token (vettore) speciale.



Packet transformer encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~
Dopo il pretrain il modello dovrebbe aver imparato ad interpretare i bytes tra di loro. Ad ogni pacchetto aggiungiamo un token speciale [PACKET] in cui il transformer ipoteticamente concentra tutte le correlazioni trovate tra pacchetti. Useremo questi embedding 'concentrati' negli step successivi.

Flow transformer encoder
~~~~~~~~~~~~~~~~~~~~~~~~
Ottenuti questi vettori rappresentanti gli interi pacchetti, 
ripassiamo per un transformer per creare correlazioni tra di loro. Ogni vettore associato ad un pacchetto otterrà una nuova rappresentazione che include le informazioni dei pacchetti circostanti.


Supplement layer (lunghezze)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per ottenere gli embedding dei pacchetti, dobbiamo troncare o allungare il pacchetto alla lunghezza desiderata dal modello transformer. Da quel punto in poi (PTE, FTE) i pacchetti trasformati avranno tutti la stessa lunghezza. Per non perdere quell'informazione, prima di fare passare per il transformer estraiamo la lunghezza di tutti i pacchetti. Si crea dunque una sequenze di queste lunghezze che viene data ad un secondo tipo di modello chiamato LSTM (long short term memory), in breve adatto ad individuare relazioni temporali all'interno di sequenze. 


Concatenazione e classificazione
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Alla fine avremo dunque:
 - Una rappresentazione dei pacchetti di un flow di traffico   trasformata per essere utile alla classificazione.
 - Una rappresentazione delle lunghezze dei pacchetti, anche   trasformata per esprimere correlazioni tra le lunghezze.
Queste due rappresentazioni vengono concatenate e data in input ad una classica rete neurale che predice il tipo di applicazione.

