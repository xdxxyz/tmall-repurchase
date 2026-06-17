package com.nju.tmall;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class Task4_PersonalRank {

    private static final double ALPHA = 0.85;
    private static final int MAX_ITER = 30;
    private static final String USER_PREFIX = "U_";
    private static final String ITEM_PREFIX = "I_";

    public static void main(String[] args) {
        if (args.length < 4) {
            System.err.println("Usage: Task4_PersonalRank <inverted_index_path> <clean_log_path> <target_user_id> <output_path>");
            System.exit(1);
        }

        String invertedIndexPath = args[0];
        String cleanLogPath      = args[1];
        String targetUserId      = args[2];
        String outputPath        = args[3];

        SparkConf conf = new SparkConf()
                .setAppName("Task4_PersonalRank_Optimized")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.kryo.registrationRequired", "false");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setCheckpointDir("hdfs:///user/" + System.getProperty("user.name", "231220104a") + "/spark-checkpoint");

        try {
            /* ========== 1) 构建邻接表 ========== */
            JavaPairRDD<String, List<String>> graph =
                    sc.textFile(invertedIndexPath)
                      .filter(l -> l != null && !l.trim().isEmpty())
                      .flatMapToPair(line -> {
                          String[] parts = line.split("\t");
                          if (parts.length < 2) return Collections.emptyIterator();
                          String itemNode = ITEM_PREFIX + parts[0].trim();
                          String[] users = parts[1].split(",");
                          List<Tuple2<String, String>> out = new ArrayList<>(users.length * 2);
                          for (String u : users) {
                              String userNode = USER_PREFIX + u.trim();
                              out.add(new Tuple2<>(itemNode, userNode));
                              out.add(new Tuple2<>(userNode, itemNode));
                          }
                          return out.iterator();
                      })
                      .combineByKey(
                          v -> { List<String> list = new ArrayList<>(); list.add(v); return list; },
                          (list, v) -> { list.add(v); return list; },
                          (list1, list2) -> { list1.addAll(list2); return list1; }
                      )
                      .cache();

            /* ========== 2) 目标用户交互历史 ========== */
            Set<String> interactedSet = new HashSet<>(
                    sc.textFile(cleanLogPath)
                       .filter(l -> l != null && l.startsWith(targetUserId + ","))
                       .map(l -> ITEM_PREFIX + l.split(",")[1].trim())
                       .collect()
            );
            Broadcast<Set<String>> bcInteracted = sc.broadcast(interactedSet);

            /* ========== 3) 初始化 PR 向量 ========== */
            final String targetNode = USER_PREFIX + targetUserId;
            JavaPairRDD<String, Double> pr =
                    graph.keys()
                         .mapToPair(n -> new Tuple2<>(n, n.equals(targetNode) ? 1.0 : 0.0))
                         .reduceByKey(Double::sum)
                         .cache();

            /* ========== 4) PersonalRank 迭代 ========== */
            for (int iter = 0; iter < MAX_ITER; iter++) {
                JavaPairRDD<String, Double> contributions =
                    pr.join(graph)
                      .flatMapToPair(kv -> {
                          String node = kv._1();
                          double prVal = kv._2()._1();
                          List<String> neighbors = kv._2()._2();
                          int deg = neighbors.size();
                          if (deg == 0) return Collections.emptyIterator();
                          double share = prVal / deg;
                          List<Tuple2<String, Double>> out = new ArrayList<>(deg);
                          for (String nb : neighbors) {
                              out.add(new Tuple2<>(nb, share));
                          }
                          return out.iterator();
                      });

                JavaPairRDD<String, Double> newPr =
                    contributions.mapValues(s -> ALPHA * s)
                                 .union(sc.parallelizePairs(
                                     Collections.singletonList(new Tuple2<>(targetNode, 1.0 - ALPHA))
                                 ))
                                 .reduceByKey(Double::sum);

                if (iter % 5 == 0) {
                    newPr.checkpoint();
                }
                newPr.cache();
                newPr.take(1);
                pr.unpersist();
                pr = newPr;
            }

            /* ========== 5) Top-K 推荐（RDD级排序，避免Driver OOM）========== */
            List<Tuple2<Double, String>> topItems =
                pr.filter(kv -> kv._1().startsWith(ITEM_PREFIX))
                  .filter(kv -> !bcInteracted.value().contains(kv._1()))
                  .mapToPair(kv -> new Tuple2<>(kv._2(), kv._1()))
                  .sortByKey(false)
                  .take(10);

            StringBuilder sb = new StringBuilder();
            for (Tuple2<Double, String> t : topItems) {
                if (sb.length() > 0) sb.append(",");
                sb.append(t._2().replace(ITEM_PREFIX, ""))
                  .append(":")
                  .append(String.format(Locale.US, "%.3f", t._1()));
            }

            String resultLine = targetUserId + "\t" + sb;
            System.out.println("✅ Task4 推荐结果：" + resultLine);

            /* ========== 6) HDFS流式写入（所有IO操作安全包裹）========== */
            try {
                org.apache.hadoop.fs.FileSystem hdfs =
                    org.apache.hadoop.fs.FileSystem.get(sc.hadoopConfiguration());
                org.apache.hadoop.fs.Path outP = new org.apache.hadoop.fs.Path(outputPath);

                if (hdfs.exists(outP)) {
                    boolean deleted = hdfs.delete(outP, true);
                    System.out.println("🗑️ 清理已有输出路径: " + outputPath + " | 结果: " + deleted);
                }

                try (OutputStream out = hdfs.create(outP, true)) {
                    out.write((resultLine + "\n").getBytes(StandardCharsets.UTF_8));
                }
                System.out.println("✅ 结果已写入: " + outputPath);
            } catch (IOException e) {
                System.err.println("❌ HDFS写入失败: " + e.getMessage());
                e.printStackTrace(System.err);
                throw e;
            }

        } catch (Throwable t) {
            System.err.println("❌ Task4 致命异常: " + t.getClass().getName() + ": " + t.getMessage());
            t.printStackTrace(System.err);
            // 不再re-throw IOException以外的异常，避免sc.stop()被跳过
            if (!(t instanceof IOException)) {
                throw new RuntimeException(t);
            }
        } finally {
            sc.stop();
        }
    }
}
