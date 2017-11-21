package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AISPipelineBridgeTest extends SparkTestBase {
	@Test
	public void testPipeline() {
		//prepare data
		JavaRDD<Row> rdd = jsc.parallelize(Arrays.asList(
				RowFactory.create(1, "string1", "string2"),
				RowFactory.create(2, "first part of string", "second part of string")
		));

		StructType schema = new StructType(new StructField[]{
				new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
				new StructField("addressLine1", DataTypes.StringType, true, Metadata.empty()),
				new StructField("addressLine2", DataTypes.StringType, true, Metadata.empty())
		});
		Dataset<Row> trainingData = spark.createDataFrame(rdd, schema);

		//train model in spark
		StringMerge stringMerge = new StringMerge()
				.setInputCol1("addressLine1")
				.setInputCol2("addressLine2")
				.setOutputCol("mergedAddress");
		//Export this model

		StringSanitizer stringSanitizer = new StringSanitizer()
				.setInputCol(stringMerge.getOutputCol())
				.setOutputCol("sanitizedAddress");

		CommonAddressFeatures commonAddressFeatures = new CommonAddressFeatures()
				.setInputCol(stringSanitizer.getOutputCol())
				.setRawInputCol(stringMerge.getOutputCol());

		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[]{stringMerge, stringSanitizer, commonAddressFeatures});

		PipelineModel pipelineModel = pipeline.fit(trainingData);

		//Export this model
		byte[] exportedModel = ModelExporter.export(pipelineModel);
		System.out.println(new String(exportedModel));

		Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

		//prepare data
		JavaRDD<Row> testRdd = jsc.parallelize(Arrays.asList(
				RowFactory.create(1, "string1", "string2"),
				RowFactory.create(2, "first part of string", "second part of string")
		));

		StructType testSchema = new StructType(new StructField[]{
				new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
				new StructField("addressLine1", DataTypes.StringType, true, Metadata.empty()),
				new StructField("addressLine2", DataTypes.StringType, true, Metadata.empty())
		});

		Dataset<Row> testData = spark.createDataFrame(testRdd, testSchema);

		List<Row> predictions = pipelineModel.transform(testData).select("id", "addressLine1", "addressLine2", "mergedAddress", "sanitizedAddress").collectAsList();

//		for (Row r : predictions) {
//			System.out.println(r);
//			String output1 = r.getString(3);
//			Map<String, Object> data = new HashMap<String, Object>();
//			data.put("addressLine1", r.getString(1));
//			data.put("addressLine2", r.getString(2));
//			transformer.transform(data);
//			String output11 = (String) data.get("mergedAddress");
//			Assert.assertEquals("output should be same", output1, output11);
//		}
	}
}
