

#python translate.py -model  -output  -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0
#python translate.py -model defaultmodel300_80/model_acc_49.08_ppl_14.86_e10.pt -output predfiles_b1/default_e10.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model defaultmodel300_80/model_acc_49.66_ppl_14.41_e15.pt -output predfiles_b1/default_e15.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model defaultmodel300_80/model_acc_47.33_ppl_17.19_e5.pt -output predfiles_b1/default_e5.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model defaultmodel300_80/model_acc_48.69_ppl_15.35_e8.pt -output predfiles_b1/default_e8.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1




#python translate.py -model model_k30a.2/model_acc_48.36_ppl_9.62_e10.pt -output predfiles_b1/local_k30_a2_e10.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model model_k30a.2/model_acc_48.86_ppl_9.35_e15.pt -output predfiles_b1/local_k30_a2_e15.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model model_k30a.2/model_acc_47.03_ppl_10.44_e5.pt -output predfiles_b1/local_k30_a2_e5.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model model_k30a.2/model_acc_47.97_ppl_9.75_e8.pt -output predfiles_b1/local_k30_a2_e8.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1

#python translate.py -model model_k30a.8/model_acc_45.55_ppl_2.00_e10.pt -output predfiles/local_k30_a8_e10.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 5

python translate.py -model model_k30a.7/model_acc_47.93_ppl_2.61_e10.pt -output predfiles_b1/local_k30_a7_e10.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
python translate.py -model model_k30a.7/model_acc_48.59_ppl_2.58_e15.pt -output predfiles_b1/local_k30_a7_e15.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
python translate.py -model model_k30a.7/model_acc_47.93_ppl_2.61_e10.pt -output predfiles/local_k30_a7_e10.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 5
python translate.py -model model_k30a.7/model_acc_48.59_ppl_2.58_e15.pt -output predfiles/local_k30_a7_e15.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 5


#python translate.py -model maxloss_model_k30a.2/model_acc_49.19_ppl_9.08_e10.pt -output predfiles_b1/max_k30_a2_e10 -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model maxloss_model_k30a.2/model_acc_49.59_ppl_8.99_e15.pt -output predfiles_b1/max_k30_a2_e15 -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1




#python translate.py -model maxloss_model_k30a.5/model_acc_48.00_ppl_4.35_e20.pt -output predfiles_b1/max_k30_a5_e20.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model maxloss_model_k30a.5/model_acc_48.18_ppl_4.34_e25.pt -output predfiles_b1/max_k30_a2_e25.txt -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model maxloss_model_k30a.5/model_acc_48.60_ppl_4.31_e10.pt -output predfiles_b1/max_k30_a5_e10 -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model maxloss_model_k30a.5/model_acc_48.96_ppl_4.27_e15.pt -output predfiles_b1/max_k30_a2_e15 -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model maxloss_model_k30a.5/model_acc_46.18_ppl_4.64_e5.pt -output predfiles_b1/max_k30_a5_e5 -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1
#python translate.py -model maxloss_model_k30a.5/model_acc_48.15_ppl_4.38_e8.pt -output predfiles_b1/max_k30_a5_e8 -src /shared/summ_data/data300_80/test.src -share_vocab -max_length 80 -replace_unk -gpu 0 -beam_size 1




