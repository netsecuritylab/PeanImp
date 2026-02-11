Argomenti di configurazione
===========================

La maggior parte delle variabili e argomenti del modello vengono gestiti con la libreria ``argparse`` che dovrebbe fare parsing degli argomenti passati da linea di comando. In pratica si esegue il codice facendo solo ``python main.py`` e si lascia che gli argomenti attesi da terminale abbiano il loro valore di default.

.. code-block:: text

  usage: main.py [-h] [--pad_num PAD_NUM] [--pad_len PAD_LEN]
                 [--pad_len_seq PAD_LEN_SEQ] [--emb EMB]
                 [--device DEVICE] [--load LOAD]
                 [--batch BATCH] [--feature FEATURE]
                 [--method METHOD] [--embway EMBWAY]
                 [--imploss IMPLOSS] [--lr LR]
                 [--length_emb_size LENGTH_EMB_SIZE]
                 [--lenhidden LENHIDDEN]
                 [--embhidden EMBHIDDEN] [--seed SEED]
                 [--trf_heads TRF_HEADS]
                 [--trf_layers TRF_LAYERS] [--mode MODE]
                 [--k K] [--epoch EPOCH]

  Traffic Classification

  options:
    -h, --help            show this help message and exit
    --pad_num PAD_NUM     the padding size of packet num
    --pad_len PAD_LEN     the padding size(length) of each
                          packet
    --pad_len_seq PAD_LEN_SEQ
                          the padding size of packet length
                          sequence
    --emb EMB             the emb size of bytes
    --device DEVICE       the training device
    --load LOAD           whether train on previous model
    --batch BATCH         batch_size
    --feature FEATURE     length / raw / ensemble
    --method METHOD       lstm / trf (Sequential Layer)
    --embway EMBWAY       random / pretrain (for raw)
    --imploss IMPLOSS     whether to use improved loss
    --lr LR               learning rate
    --length_emb_size LENGTH_EMB_SIZE
                          len emb size
    --lenhidden LENHIDDEN
                          len hidden size
    --embhidden EMBHIDDEN
                          emb hidden size
    --seed SEED           random seed
    --trf_heads TRF_HEADS
                          transformers heads number
    --trf_layers TRF_LAYERS
                          transformers layers
    --mode MODE           train/test
    --k K                 k fold validation
    --epoch EPOCH         epoch

.. data:: block_size
  
  Indica il numero di token massimo che ogni volta vengono dati in input al modello.

.. data:: output_dir

   Dove vengono salvati i pesi allenati e i report dell'evaluation dei modelli, per default è `/Model/.`
