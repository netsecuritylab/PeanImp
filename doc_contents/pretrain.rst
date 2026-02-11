Pretraining
===========
Nel pretrain viene allenato il modello ad 'interpretare' i byte cifrati di un pacchetto. Questa capacità interpretativa verrà riutilizzata negli step successivi dell'allenamento complessivo. L'allenamento avviene attraverso il **Masked Language Modeling** (MLM), dove essenzialmente verrà data una sequenza di bytes al modello, alcuni dei quali saranno 'mascherati' e l'obiettivo del transformer sarà di ricostruirli.

.. function:: train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]

    Questa è la funzione principale per allenare il modello. Questo è un link ad essa :func:`train`

Mascherare i bytes
------------------------

.. function:: mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]

   Questa è la funzione usata per mascherare i tokens subit prima di darli in input al modello.

Padding
-------



Dataset
-------

Vedi l'argomento :data:`block_size`. 

.. class:: TextDataset(Dataset)
   
   Si aspetta un dataset in cui byte è separato da uno spazio, siccome viene usato durante il pretrain non ci importa a quale pacchetto i byte appartengano. Di ogni linea prenderà un tot di bytes e quello diventerà un sample per il modello.
   Carica il dataset dal filepath specificato attraverso pickle. Si aspetta un dataset già tokenizzato. Se non esiste, crea una cache dove viene salvato il testo in input con i token sostituiti dai loro id e con gli special tokens già inseriti, ottimizzando le performance. 

.. class:: LineByLineTextDataset(Dataset)

   Apre il filepath specificato da dataset e tratta ogni riga del file di testo indicato come un flow. A differenza di :class:`TextDataset` non salva in una cache il testo già tokenizzato.

Usare i file pcap
~~~~~~~~~~~~~~~~~

Siccome :class:`TextDataset` richiede che i byte siano separati da uno spazio, bisogna fare del veloce preprocessing per leggere i .pcap e scrivere i bytes in un file di testo .txt. Per fare questo parsing ho usato la libreria `scapy` (assieme a `binascii`) e la funzione `readPcap` che fa il parsing si trova in `pretrain/preTrainUtils.py`.
