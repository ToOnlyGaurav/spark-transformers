package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.CommonAddressFeaturesModelInfo;
import org.apache.spark.ml.CommonAddressFeatures;

import java.util.Arrays;
import java.util.HashSet;


public class CommonAddressFeaturesModelInfoAdapter extends AbstractModelInfoAdapter<CommonAddressFeatures, CommonAddressFeaturesModelInfo> {
	@Override
	CommonAddressFeaturesModelInfo getModelInfo(CommonAddressFeatures from) {
		CommonAddressFeaturesModelInfo modelInfo = new CommonAddressFeaturesModelInfo();
		modelInfo.setFavourableStarts(new HashSet<>(Arrays.asList(from.favourableStartWords())));
		modelInfo.setUnFavourableStarts(new HashSet<>(Arrays.asList(from.unfavourableStartWords())));

		modelInfo.setSanitizedAddressParam(from.getInputCol());
		modelInfo.setMergedAddressParam(from.getRawInputCol());

		modelInfo.setNumWordsParam(from.getNumWordsParam());
		modelInfo.setNumCommasParam(from.getNumCommasParams());
		modelInfo.setNumericPresentParam(from.getNumericPresentParam());
		modelInfo.setAddressLengthParam(from.getAddressLengthParam());
		modelInfo.setFavouredStartColParam(from.getFavouredStartColParam());
		modelInfo.setUnfavouredStartColParam(from.getUnfavouredStartColParam());

		return modelInfo;
	}

	@Override
	public Class<CommonAddressFeatures> getSource() {
		return CommonAddressFeatures.class;
	}

	@Override
	public Class<CommonAddressFeaturesModelInfo> getTarget() {
		return CommonAddressFeaturesModelInfo.class;
	}
}
