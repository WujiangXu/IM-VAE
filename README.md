# **Instruction**

This anonymous code repository is for the paper "Information Maximization Variational Autoencoder for Cross-Domain Sequential Recommendation".

# **Quick Run**

We provide an example terminal command line in the "run.sh" bash file. To run the code, simply execute the bash file in the terminal:

```bash
sh run.sh
```

Alternatively, you can copy the command line from "run.sh" and run it directly in the terminal:

```bash
python train_our.py -m "amazon_results/Cloth_Sport_Ours/ours+2vae+KLD0.005+KLD20.001_8e-4_AddR+CS_O" -dm cloth_sport  --kl_lambda_1 0.005 --kl_lambda_2 0.005 --kl_lambda_1_t 0.001 --kl_lambda_2_t 0.001 --lr 8e-4 --KLD1 1.5 --trans_encoder 'attention' --cs_setting True &
```

In the command line above:

* "-m" specifies the file path for saving training and prediction results.
* "-dm" refers to the dataset.
* The KL hyper-parameters refer to the $\lambda_a$ and $\lambda_t$ for domain 1 and 2.
* "--trans_encoder" specifies the type of model used by the cross-domain encoder (choose from "mlp" and "attention").
* "--cs_setting" refers to whether use infererence variants of $r^x$/$r^y$ for cold-start users.

