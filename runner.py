import sys
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging

# إضافة مسار الكود
sys.path.append("/home/ubuntu/projects/project-8d89238b/extracted")
from logic import MahwousEngine, SemanticIndex, load_store_products, load_competitor_products, load_brands, export_salla_csv

# إعداد السجلات
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

def run_golden_test():
    # 1. تحميل البيانات
    store_file = "/home/ubuntu/upload/متجرنامهووسبكلالاعمدةللمنتجات.csv"
    comp_files = [
        "/home/ubuntu/upload/متجرساراءميكاب.csv",
        "/home/ubuntu/upload/متجرعالمجيفنشيبكلالاعمدةالسعروالصور.csv"
    ]
    brands_file = "/home/ubuntu/upload/ماركاتمهووس.csv"

    print("--- Loading Data ---")
    store_df = load_store_products([store_file])
    comp_df = load_competitor_products(comp_files)
    brands = load_brands(brands_file)
    
    print(f"Store products: {len(store_df)}")
    print(f"Competitor products: {len(comp_df)}")
    print(f"Brands loaded: {len(brands)}")

    # 2. بناء الفهرس الدلالي
    print("\n--- Building Semantic Index ---")
    model = SentenceTransformer(SemanticIndex.MODEL_NAME)
    idx = SemanticIndex(model)
    idx.build(store_df)

    # 3. تشغيل المحرك (Golden Match v9.0 with LLM)
    print("\n--- Running Golden Match Engine v9.0 with LLM ---")
    engine = MahwousEngine(semantic_index=idx, brands_list=brands)
    new_opps, duplicates, reviews = engine.run(store_df, comp_df, use_llm=True)

    # 4. طباعة النتائج النهائية
    print("\n--- Final Results Summary ---")
    print(f"New Opportunities: {len(new_opps)}")
    print(f"Duplicates Blocked: {len(duplicates)}")
    print(f"Manual Reviews (Gray Zone): {len(reviews)}")

    # 5. حفظ النتائج وتصدير ملف سلة
    pd.DataFrame([vars(r) for r in new_opps]).to_csv("golden_new_opps.csv", index=False)
    pd.DataFrame([vars(r) for r in duplicates]).to_csv("golden_duplicates.csv", index=False)
    pd.DataFrame([vars(r) for r in reviews]).to_csv("golden_reviews.csv", index=False)
    
    salla_csv = export_salla_csv(new_opps)
    with open("salla_golden_opportunities.csv", "wb") as f:
        f.write(salla_csv)
    
    print("\n[SUCCESS] Final Golden Files Generated.")

if __name__ == "__main__":
    run_golden_test()
