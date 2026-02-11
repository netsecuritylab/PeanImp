.. PEAN docs documentation master file, created by
   sphinx-quickstart on Mon Dec  8 11:01:50 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentazione PEAN
=====================================
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   doc_contents/pretrain

   doc_contents/configuration

   doc_contents/how_it_works

   doc_contents/model_args

Come funziona il modello
------------------------

Il modello è diviso in 4 fasi diverse, che sono in ordine:
 - Pretrain
 - Packet tranfsormer encoder (PTE)
 - Flow transformer encoder (FTE)
 - Supplement layer (lunghezze pacchetti) 
 - Concatenazione e classificazione

Ogni step è spiegato nel dettaglio in :doc:`doc_contents/how_it_works`.

Come vengono passati i dati al modello
--------------------------------------

Il dataset che il modello si attende è già diviso in traffici corrispondenti alla stessa applicazione ovvero: ogni sample estratto dal dataset contiene 10 pacchetti (ognuno troncato per avere 400 bytes) dello stesso tipo di traffico. 

Questo lavoro di catalogazione non è presente all'interno del codice fornito dagli autori. 


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
