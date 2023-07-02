#for data in 1 2  4 5
#do
python InceptionTimeFuse.py\
	--data_root=/home/getch/ssl/EO_Africa/samples\
    	--weight_path=/home/getch/ssl/EO_Africa/MODELS\
    	--log_path=/home/getch/ssl/EO_Africa/LOGS\
    	--index_fold=/home/getch/ssl/EO_Africa/FILL\
    	--pred_scene_fold=/home/getch/ssl/EO_Africa/PREDS\
    	--scene_fold=/home/getch/ssl/EO_Africa/FILL\
	--feature_fold=/home/getch/ssl/EO_Africa/FEATURES\
    	--raster_tempelate=/home/getch/ssl/EO_Africa/plntMask.tif\
    	--hdims=128\
    	--nlayer=4\
	--dropout=0.15\
    	--batch_size=32\
    	--lr=0.0001\
    	--epochs=50\
    	--runner=train_test\
    	--model=inceptionTime\
    	--data=fused_deep\

#done 
