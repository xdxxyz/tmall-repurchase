package com.nju.tmall;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import scala.Tuple2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Task1_CleanAndFeature implements Serializable {

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println(
                "Usage: spark-submit --class com.nju.tmall.Task1_CleanAndFeature <jar> " +
                "<input> <cleanOutput> <featureOutput>"
            );
            System.exit(1);
        }
        run(args[0], args[1], args[2]);
    }

    public static void run(String inputPath, String cleanOut, String featureOut) {

        // ========= 1) Spark 启动 =========
        SparkConf conf = new SparkConf().setAppName("Task1_CleanAndFeature");
        // conf.setMaster("local[*]"); // 本地 IDEA 调试才开
        JavaSparkContext sc = new JavaSparkContext(conf);

        // ========= 2) 删除旧输出目录（关键：防止 FileAlreadyExistsException）=========
        try {
            Configuration hConf = sc.hadoopConfiguration();  // 自动拿到 hdfs/hadoop 配置
            FileSystem fs = FileSystem.get(hConf);
            deleteIfExists(fs, cleanOut);
            deleteIfExists(fs, featureOut);
        } catch (Exception e) {
            // 删除失败通常不是致命的，这里不打断主流程；真·路径/权限问题会在 saveAsTextFile 才暴露
            System.err.println("WARN: clean output dir failed: " + e.getMessage());
        }

        // ========= 3) 读文件 + 清洗 =========
        JavaPairRDD<Long, Rec> valid = sc.textFile(inputPath)
            .flatMapToPair(line -> {
                List<Tuple2<Long, Rec>> out = new ArrayList<>(1);
                String l = line.trim();
                if (l.isEmpty()) return out.iterator();

                // 跳表头
                if (l.contains("user_id") && l.contains("timestamp")) return out.iterator();
                if (l.startsWith("user_id") || l.startsWith("timestamp"))
                    return out.iterator();

                // 兼容 逗号 / 空白分隔
                String[] p;
                if (l.indexOf(',') >= 0) p = l.split(",", -1);
                else                     p = l.split("\\s+", -1);

                if (p.length != 5) return out.iterator();

                String bt = p[3].trim();
                if (!isValidBehavior(bt)) return out.iterator();

                try {
                    long uid = Long.parseLong(p[0].trim());
                    long iid = Long.parseLong(p[1].trim());
                    long cid = Long.parseLong(p[2].trim());
                    long ts  = Long.parseLong(p[4].trim());
                    out.add(new Tuple2<>(uid, new Rec(uid, iid, cid, bt, ts)));
                } catch (NumberFormatException ignored) {}
                return out.iterator();
            });

        // ========= 4) 输出 clean_behavior_log（清洗后日志）==========
        valid.map(t -> {
            Rec r = t._2;
            return r.uid + "," + r.iid + "," + r.cid + "," + r.bt + "," + r.ts;
        }).saveAsTextFile(cleanOut);

        // ========= 5) 全局截止时间 maxTs = 全量最大时间戳 =========
        final long maxTs = valid.map(t -> t._2.ts).reduce(
            (Function2<Long, Long, Long>) (a, b) -> a > b ? a : b
        );

        // ========= 6) aggregateByKey：per-user → lastTs / cnt / scoreSum =========
        JavaPairRDD<Long, Agg> userAgg = valid
            .mapToPair(t -> new Tuple2<>(t._1, t._2))
            .aggregateByKey(
                new Agg(),
                (agg, r) -> {
                    if (r.ts > agg.lastTs) agg.lastTs = r.ts;
                    agg.cnt++;
                    agg.scoreSum += scoreOf(r.bt);
                    return agg;
                },
                (a, b) -> {
                    Agg r = new Agg();
                    r.lastTs   = Math.max(a.lastTs, b.lastTs);
                    r.cnt      = a.cnt + b.cnt;
                    r.scoreSum = a.scoreSum + b.scoreSum;
                    return r;
                }
            );

        // ========= 7) 归一化 Min-Max =========
        List<Tuple2<Long, double[]>> rows = userAgg.map(t -> {
            Agg a = t._2;
            double f1 = (maxTs - a.lastTs) / 86400.0;
            double f2 = (double) a.cnt;
            double f3 = (double) a.scoreSum;
            return new Tuple2<>(t._1, new double[]{f1, f2, f3});
        }).collect();

        double f1_min = Double.MAX_VALUE, f1_max = Double.MIN_VALUE;
        double f2_min = Double.MAX_VALUE, f2_max = Double.MIN_VALUE;
        double f3_min = Double.MAX_VALUE, f3_max = Double.MIN_VALUE;
        for (Tuple2<Long, double[]> x : rows) {
            double f1 = x._2[0], f2 = x._2[1], f3 = x._2[2];
            if (f1 < f1_min) f1_min = f1; if (f1 > f1_max) f1_max = f1;
            if (f2 < f2_min) f2_min = f2; if (f2 > f2_max) f2_max = f2;
            if (f3 < f3_min) f3_min = f3; if (f3 > f3_max) f3_max = f3;
        }
        double f1_range = (f1_max - f1_min) < 1e-12 ? 1.0 : (f1_max - f1_min);
        double f2_range = (f2_max - f2_min) < 1e-12 ? 1.0 : (f2_max - f2_min);
        double f3_range = (f3_max - f3_min) < 1e-12 ? 1.0 : (f3_max - f3_min);

        List<String> outLines = new ArrayList<>();
        for (Tuple2<Long, double[]> x : rows) {
            long uid = x._1;
            double nf1 = clamp01((x._2[0] - f1_min) / f1_range);
            double nf2 = clamp01((x._2[1] - f2_min) / f2_range);
            double nf3 = clamp01((x._2[2] - f3_min) / f3_range);
            outLines.add(
                uid + "," +
                String.format("%.4f", nf1) + "," +
                String.format("%.4f", nf2) + "," +
                String.format("%.4f", nf3)
            );
        }

        // ========= 8) 写特征输出 =========
        sc.parallelize(outLines, 1).saveAsTextFile(featureOut);

        sc.stop();
    }

    // ======== 安全删除（不存在也不抛）========
    private static void deleteIfExists(FileSystem fs, String p) throws Exception {
        Path path = new Path(p);
        if (fs.exists(path)) {
            fs.delete(path, true);
        }
    }

    // ======== 行为合法性 / 分值 ========
    static boolean isValidBehavior(String bt) {
        return "pv".equals(bt) || "cart".equals(bt) || "fav".equals(bt) || "buy".equals(bt);
    }

    static int scoreOf(String bt) {
        switch (bt) {
            case "buy":  return 4;
            case "cart": return 3;
            case "fav":  return 2;
            default:     return 1; // pv
        }
    }

    static double clamp01(double v) {
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        return v;
    }

    // ======== POJO ========
    static class Rec implements Serializable {
        long uid, iid, cid, ts;
        String bt;
        Rec(long uid, long iid, long cid, String bt, long ts) {
            this.uid = uid; this.iid = iid; this.cid = cid; this.bt = bt; this.ts = ts;
        }
    }

    static class Agg implements Serializable {
        long lastTs = 0;
        long cnt = 0;
        long scoreSum = 0;
    }
}
