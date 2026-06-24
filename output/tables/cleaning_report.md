# Отчёт об очистке данных

Исходных строк: 24055

Удалены почти пустые столбцы: ['heating_type', 'ceiling_height_m', 'parking_type', 'passenger_lifts_count']
Удалены неиспользуемые столбцы (тексты/идентификаторы/константы): ['url', 'title', 'description', 'detail_features', 'image_urls', 'image_paths', 'images_count', 'deal_type', 'price_formatted', 'repair', 'address', 'residential_complex', 'loggias_count', 'balconies_count', 'combined_wcs_count', 'separate_wcs_count', 'pets_allowed', 'children_allowed', 'utilities_included', 'details_fetched']

`rooms`: 3082 пропусков импутированы как 0 (студия), добавлен флаг `rooms_was_missing`.

Удалены строки без станции метро: 68 (0.3% датасета).

`living_area_sqm`: пропуски заменены медианой по группе `rooms`, добавлен флаг `living_area_sqm_missing`.

`kitchen_area_sqm`: пропуски заменены медианой по группе `rooms`, добавлен флаг `kitchen_area_sqm_missing`.

`build_year`: пропуски заменены медианой по группе `rooms`, добавлен флаг `build_year_missing`.

`house_material`, `repair_type`: пропуски заменены явной категорией 'unknown'.

`author_type`/`author` заменены бинарным флагом `is_agent` (только 'agent' был размечен явно).

`deposit_rub`, `prepay_months`: пропуски (~8-10%) заменены медианой, добавлены флаги missing.

Отсечены выбросы по 1-99 перцентилю `price_rub` и `area_sqm`: удалено 827 строк (3.4%).

Добавлена `log_price_rub` = log1p(price_rub) как целевая переменная для моделирования (skew price_rub=3.14, skew log_price_rub=1.13).


Итоговых строк: 23160 (было 24055, потеряно 895 = 3.7%).


Итоговые столбцы (25): ['added', 'area_sqm', 'build_year', 'build_year_missing', 'deposit_rub', 'deposit_rub_missing', 'district', 'floor', 'floors_total', 'house_material', 'is_agent', 'kitchen_area_sqm', 'kitchen_area_sqm_missing', 'living_area_sqm', 'living_area_sqm_missing', 'log_price_rub', 'metro', 'metro_time_min', 'offer_id', 'prepay_months', 'prepay_months_missing', 'price_rub', 'repair_type', 'rooms', 'rooms_was_missing']
