import asyncio
import aiohttp

# Limite le nombre de requêtes concurrentes à 50 (vous pouvez ajuster cette valeur)
semaphore = asyncio.Semaphore(50)

async def fetch_variant(session, vid, retries=3, delay=1):
    """Récupère l'annotation pour un variant donné, avec réessais en cas d'échec."""
    url = f"http://myvariant.info/v1/variant/{vid}"
    # Limiter l'accès à la requête via le sémaphore
    async with semaphore:
        for attempt in range(1, retries + 1):
            async with session.get(url) as response:
                data = await response.json()
                # Si "success" existe et vaut False, on réessaie après un délai
                if data.get("success", True) is False:
                    print(f"Attempt {attempt} for {vid} failed (success=False). Retrying after {delay} sec...")
                    await asyncio.sleep(delay)
                    continue
                return vid, data
        # Après tous les essais, retourner le dernier résultat (même s'il est marqué comme échec)
        return vid, data

async def fetch_all_variants(variant_ids):
    variant_data = {}
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_variant(session, vid) for vid in variant_ids]
        for i, future in enumerate(asyncio.as_completed(tasks), 1):
            vid, data = await future
            variant_data[vid] = data
            if i % 10 == 0:
                print(f"Processed {i} variants")
    return variant_data

if __name__ == '__main__':
    import pandas as pd
    import json

    # Charger les données et générer les variant_ids
    maf_df = pd.read_csv("data/X_train/molecular_train.csv")
    maf_eval = pd.read_csv("data/X_test/molecular_test.csv")

    maf_df = pd.concat([maf_df, maf_eval])
    maf_df = maf_df.dropna(subset=['CHR', 'START', 'REF', 'ALT'])

    def make_variant_id(row):
        return f"chr{row['CHR']}:g.{int(row['START'])}{row['REF']}>{row['ALT']}"

    maf_df['variant_id'] = maf_df.apply(make_variant_id, axis=1)
    maf_df = maf_df[~maf_df['variant_id'].str.contains("nan")]

    variant_ids = maf_df['variant_id'].unique()

    # Exécution asynchrone pour récupérer les annotations
    variant_data = asyncio.run(fetch_all_variants(variant_ids))

    # Enregistrer les données dans un fichier JSON
    with open("data/variant_data.json", "w") as f:
        json.dump(variant_data, f)