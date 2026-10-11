from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T

BASE = Path(__file__).resolve().parent

spark = SparkSession.builder.appName("DE_3").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


def step(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# 1. Завантажуємо CSV в окремі DataFrame. Явні схеми замість inferSchema: дата читається як DateType, а ціна як Decimal, щоб суми грошей рахувалися точно, без похибок float.
users_schema = T.StructType([
    T.StructField("user_id", T.IntegerType()),
    T.StructField("name", T.StringType()),
    T.StructField("age", T.IntegerType()),
    T.StructField("email", T.StringType()),
])
purchases_schema = T.StructType([
    T.StructField("purchase_id", T.IntegerType()),
    T.StructField("user_id", T.IntegerType()),
    T.StructField("product_id", T.IntegerType()),
    T.StructField("date", T.DateType()),
    T.StructField("quantity", T.IntegerType()),
])
products_schema = T.StructType([
    T.StructField("product_id", T.IntegerType()),
    T.StructField("product_name", T.StringType()),
    T.StructField("category", T.StringType()),
    T.StructField("price", T.DecimalType(10, 2)),
])


def read_csv(name: str, schema: T.StructType):
    return (spark.read
            .option("header", True)
            .option("mode", "PERMISSIVE")   # неповні рядки -> null, а не помилка
            .schema(schema)
            .csv(str(BASE / f"{name}.csv")))


users_raw = read_csv("users", users_schema)
purchases_raw = read_csv("purchases", purchases_schema)
products_raw = read_csv("products", products_schema)

step("1. Завантажені дані")
for name, df in [("users", users_raw), ("purchases", purchases_raw),
                 ("products", products_raw)]:
    print(f"\n--- {name}: {df.count()} рядків ---")
    df.show(5, truncate=False)


# 2. Очищення: видалямо рядкі з будь-якими пропущеними значеннями
users = users_raw.dropna(how="any")
purchases = purchases_raw.dropna(how="any")
products = products_raw.dropna(how="any")

step("2. Очищення даних (dropna)")
print("\nВидалені рядки з пропусками:")
for name, df in [("users", users_raw), ("purchases", purchases_raw),
                 ("products", products_raw)]:
    cond = " OR ".join(f"`{c}` IS NULL" for c in df.columns)
    print(f"\n--- {name} ---")
    df.filter(cond).show(truncate=False)

stats = spark.createDataFrame(
    [(n, raw.count(), clean.count())
     for n, raw, clean in [("users", users_raw, users),
                           ("purchases", purchases_raw, purchases),
                           ("products", products_raw, products)]],
    "table string, rows_before long, rows_after long",
).withColumn("removed", F.col("rows_before") - F.col("rows_after"))
print("Підсумок очищення:")
stats.show(truncate=False)

# Спільна "факт-таблиця": покупка + продукт + вартість рядка покупки. При цьомуInner join відкидає покупки, чиї продукти було видалено при очищенні.
sales = (purchases
         .join(products, on="product_id", how="inner")
         .withColumn("total", F.col("quantity") * F.col("price")))

# 3. Загальна сума покупок за кожною категорією
total_by_category = (sales
                     .groupBy("category")
                     .agg(F.round(F.sum("total"), 2).alias("total_amount"))
                     .orderBy(F.desc("total_amount")))

step("3. Загальна сума покупок за категоріями")
total_by_category.show(truncate=False)

# 4. Сума покупок за категоріями для віку 18–25 включно
sales_18_25 = (sales
               .join(users, on="user_id", how="inner")
               .filter(F.col("age").between(18, 25)))   # between — включно

total_18_25 = (sales_18_25
               .groupBy("category")
               .agg(F.round(F.sum("total"), 2).alias("total_amount"))
               .orderBy(F.desc("total_amount")))

step("4. Сума покупок за категоріями, вік 18–25")
total_18_25.show(truncate=False)

# 5. Частка кожної категорії від сумарних витрат групи 18–25 (%)
grand_total = total_18_25.agg(F.sum("total_amount").alias("grand_total"))

share_18_25 = (total_18_25
               .crossJoin(grand_total)
               .withColumn("percentage",
                           F.round(F.col("total_amount") / F.col("grand_total") * 100, 2))
               .select("category", "total_amount", "percentage")
               .orderBy(F.desc("percentage")))

step("5. Частка витрат за категоріями, вік 18–25 (%)")
share_18_25.show(truncate=False)

# 6. Топ-3 категорії за відсотком витрат, вік 18–25
top3 = share_18_25.limit(3)

step("6. Топ-3 категорії за відсотком витрат, вік 18–25")
top3.show(truncate=False)

spark.stop()
