package com.nju.tmall;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.*;

public class Task2_Analysis {

    /** clean_out 行格式（Task1 产出）:
     *  user_id,item_id,category_id,behavior_type,timestamp
     *
     *  args[0] = hdfs://.../clean_out   (目录)
     *  args[1] = hdfs://.../task2_out    (输出基目录)
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println(
                "Usage: spark-submit --class com.nju.tmall.Task2_Analysis <jar> " +
                "<cleanOutDir> <outputBaseDir>\n" +
                "  e.g.\n" +
                "    --class com.nju.tmall.Task2_Analysis tmall-repurchase-1.0.jar \\\n" +
                "         hdfs://hcdsj/user/231220104a/clean_out \\\n" +
                "         hdfs://hcdsj/user/231220104a/task2_out"
            );
            System.exit(1);
        }

        String input  = args[0].trim();
        String output = args[1].trim();

        SparkConf conf = new SparkConf().setAppName("Task2_InvertedAndCooccur");
        // conf.setMaster("local[*]"); // ← 本地 IDEA 调试才开；集群提交注释掉
        JavaSparkContext sc = new JavaSparkContext(conf);

        /* =====================================================
         * 0) 删旧输出：防止 saveAsTextFile 炸 FileAlreadyExists
         * ===================================================== */
        try {
            Configuration hc = sc.hadoopConfiguration();
            FileSystem fs = FileSystem.get(hc);
            fs.delete(new Path(output), true);
        } catch (Exception e) {
            // 某些 HA/权限场景 delete 可能没权限，先放过；真·路径问题会在 saveAsTextFile 暴露
            System.err.println("[WARN] cannot delete old output base (may already not exist): " + e.getMessage());
        }

        String idxDir = output + "/item_user_inverted_index";
        String coDir  = output + "/item_co_occurrence";

        /* =====================================================
         * 1) 读行 + 硬防护表头/脏行
         *    分隔符是逗号（Task1 的 clean_behavior_log.csv 格式）
         * ===================================================== */
        JavaRDD<String> lines = sc.textFile(input);

        // 搜索工具：只保留“首字段看起来像纯数字 user_id”的行
        JavaRDD<String> cleanLines = lines.filter(l -> {
            String s = (l == null) ? null : l.trim();
            if (s == null || s.isEmpty()) return false;
            String[] p = s.split(",", -1);
            if (p.length < 5) return false;
            String maybeUid = p[0].trim();
            // user_id 必须是纯数字（可能带负号不要紧，但你的数据不会负；\d+ 最稳）
            return maybeUid.matches("\\d+");
        });

        /* =====================================================
         * 2) 倒排索引：item -> 去重 user 列表
         *    仅使用 深度交互 behavior ∈ {cart, fav, buy}
         * ===================================================== */
        // (item_id, user_id)
        JavaPairRDD<String, String> deepPairs = cleanLines.mapToPair(l -> {
            String[] p = l.split(",", -1);
            String uid = p[0].trim();
            String iid = p[1].trim();
            String bt  = p[3].trim();
            if (!bt.equals("cart") && !bt.equals("fav") && !bt.equals("buy"))
                return null;
            return new Tuple2<>(iid, uid);
        }).filter(t -> t != null && !t._1().isEmpty() && !t._2().isEmpty());

        // item -> "u1,u2,u3..."
        JavaRDD<String> idxLines = deepPairs
            .groupByKey()
            .map(t -> {
                String item = t._1();
                LinkedHashSet<String> us = new LinkedHashSet<>();
                t._2().forEach(us::add);
                StringBuilder sb = new StringBuilder();
                int i = 0;
                for (String u : us) {
                    if (i++ > 0) sb.append(",");
                    sb.append(u);
                }
                // 搜索工具要求格式：item_id\t u1,u2,u3...
                return item + "\t" + sb.toString();
            });

        idxLines.coalesce(1).saveAsTextFile(idxDir);

        /* =====================================================
         * 3) 共现：同用户买了多个不同 item -> 无序 pair 计数
         *    这里按 PPT 口径：仅 buy（要放宽就把下面 "buy" 改成 Set 包含 cart/fav）
         * ===================================================== */
        // (user_id, item_id)  where behavior=buy
        JavaPairRDD<String, String> buyPairs = cleanLines.mapToPair(l -> {
            String[] p = l.split(",", -1);
            if (!p[3].trim().equals("buy")) return null;
            return new Tuple2<>(p[0].trim(), p[1].trim());
        }).filter(t -> t != null && !t._1().isEmpty() && !t._2().isEmpty());

        JavaPairRDD<String, Iterable<String>> userBasket = buyPairs.groupByKey();

        // pair=(itemA,itemB) 其中 A<=B 字典序，避免 (A,B)/(B,A) 分裂
        JavaPairRDD<Tuple2<String,String>, Long> pairCount =
            userBasket.flatMapToPair(kv -> {
                LinkedHashSet<String> set = new LinkedHashSet<>();
                kv._2().forEach(set::add);
                ArrayList<String> items = new ArrayList<>(set);
                Collections.sort(items);
                ArrayList<Tuple2<Tuple2<String,String>, Long>> out = new ArrayList<>();
                for (int i = 0; i < items.size(); i++) {
                    for (int j = i + 1; j < items.size(); j++) {
                        out.add(new Tuple2<>(new Tuple2<>(items.get(i), items.get(j)), 1L));
                    }
                }
                return out.iterator();
            })
            .reduceByKey(Long::sum);

        // 显式 swap→sort→swap 回：避免 Java 编译器在 lambda 里卡 Tuple2 泛型推导
        JavaPairRDD<Long, Tuple2<String,String>> swapped =
            pairCount.mapToPair(kv -> new Tuple2<>(kv._2(), kv._1()));

        List<Tuple2<Tuple2<String,String>, Long>> top =
            swapped
                .sortByKey(false)          // count desc
                .mapToPair(kv -> new Tuple2<>(kv._2(), kv._1()))
                .take(1000);               // 前 1000 条

        JavaRDD<String> coLines = sc.parallelize(top)
            .map(e -> {
                String a = e._1()._1();
                String b = e._1()._2();
                long c  = e._2();
                // 搜索工具要求格式：item_id_A,item_id_B\t co_occurrence_count
                return a + "," + b + "\t" + c;
            });

        coLines.coalesce(1).saveAsTextFile(coDir);

        /* =====================================================
         * 4) 日志：告诉你在哪看结果（验收时截屏用）
         * ===================================================== */
        System.out.println("==== TASK2 DONE ====");
        System.out.println("  baseDir          = " + output);
        System.out.println("  inverted_index   = " + idxDir  + "  (see part-00000)");
        System.out.println("  co_occurrence    = " + coDir   + "  (see part-00000, top 1000)");

        sc.stop();
    }
}
