package com.martil.bitcoin.bot

import java.io.File

import hex.tree.gbm.GBM
import hex.tree.gbm.GBMModel.GBMParameters
import org.apache.spark.h2o.H2OContext
import org.apache.spark.{SparkFiles, SparkConf}
import org.apache.spark.sql.SparkSession
import water.fvec.H2OFrame


/**
 * A basic starter application as proof of work for Sparkling Water and Apache Spark
 */
object SparkDemo extends App {
  //create Spark context
  val appConfig = configure("Basic Starter App")
  val spark = SparkSession.builder.config(appConfig).getOrCreate()

  //create H2O context
  // FIXME: deprecated
  val h2oContext = H2OContext.getOrCreate(spark)

  spark.sparkContext.addFile("/src/main/resources/sample_data.csv")
  val dataTable = new H2OFrame(new File(SparkFiles.get("sample_data.csv")))

  // Build GBM model
  val gbmParams = new GBMParameters()
  //FIXME: deprecated
  //gbmParams._train = dataTable
  gbmParams._response_column = "class"
  gbmParams._ntrees = 5

  val gbm = new GBM(gbmParams)
  val gbmModel = gbm.trainModel.get

  // Make prediction on train data
  val predict = gbmModel.score(dataTable).subframe(Array("predict"))

  //FIXME: All deprecated and doesn't compile

  // Compute number of mispredictions with help of Spark API
//  val trainRDD = h2oContext.asRDD[StringHolder](dataTable.subframe(Array("class")))
//  val predictRDD = h2oContext.asRDD[StringHolder](predict)
//
//  // Make sure that both RDDs has the same number of elements
//  assert(trainRDD.count() == predictRDD.count)
//  val numMispredictions = trainRDD.zip(predictRDD).filter(i => {
//    val act = i._1
//    val pred = i._2
//    act.result != pred.result
//  }).collect()
//
//  println(
//    s"""
//       |Number of mispredictions: ${numMispredictions.length}
//       |
//       |Mispredictions:
//       |
//       |actual X predicted
//       |------------------
//       |${numMispredictions.map(i => i._1.result.get + " X " + i._2.result.get).mkString("\n")}
//       """.stripMargin)

  // Shutdown application
  h2oContext.stop(true)

  def configure(appName: String): SparkConf = {
    val defaultName = "spark.master"
    new SparkConf().setAppName(appName).setIfMissing(defaultName, sys.env.getOrElse(defaultName, "local"))
      .set("spark.sql.autoBroadcastJoinThreshold", "-1")
  }
}
