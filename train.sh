#for data in 1 2  4 5
#do
python InceptionTimeFuse.py\
	--data_root=./EO_Africa/samples\
    	--weight_path=./EO_Africa/MODELS\
    	--log_path=./EO_Africa/LOGS\
    	--index_fold=./EO_Africa/FILL\
    	--pred_scene_fold=./EO_Africa/PREDS\
    	--scene_fold=./EO_Africa/FILL\
	--feature_fold=./EO_Africa/FEATURES\
    	--raster_tempelate=./EO_Africa/plntMask.tif\
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
