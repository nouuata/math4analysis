# math4analysis

Настоящата папка съдържа данни и графики свързани с статията "Изследване на достъпа до разширено изучаване на математика в 5 - 7 клас и липсата на възможности пред талантливите ученици от големите градове в България".

## Съдържание

[`data`]:  Данни за математически състезания за прием в 5 клас през 2023/2024 година
- [`keng`]:    Структурирани резултати по области oт Европейско Кенгуру 2023/2024
- [`matvs`]:   Структурирани резултати по области oт Математика за Всеки 2023/2024
- [`olymp`]:   Структурирани резултати по области oт Областен кръг на Олимпиадата 2023/2024
- [`pms`]:     Структурирани резултати по области oт Пролетно Математическо Състезание 2023/2024 
- [`merged`]:  Сляти резултати на четирите състезания по области
- [`derived`]: Данни свързани с анализа
   - [`by_region`]: Данни свързани с анализа сегментирани по региони
   - [`cdf`]:       Данни свързани с кумулативните дистрибуции
   - [`classes`]:   Данни свързани с анализа на класове

[`plots`]:  Графики свързани с математически състезания за прием в 5 клас през 2023/2024 година
- [`article`]:         Графики използвани в статията
- [`by_acceptance`]:   Графики демонстриращи различни критерии за прием 
- [`plot_by_region`]:       Графики демонстриращи регионални различия в приема
- [`plot_cdf`]:             Кумулативни функции
- [`hist`]:            SGI хистограми
- [`joint_3d`]:        Съвместни 3D дистрибуции
- [`joint_dist`]:      Съвместни дистрибуции
   - [`with_size`]:        Съвместни дистрибуции с данни за големината на групата

## Генериране на данните и графиките

Генериране е нужно само в случайте, когато бъдат направени промени в скриптовете. Процесът на генерация изисква следната последователност от команди:
```
python3 merger.py 
python3 derive_data_and_plots.py
python3 derive_article_plots.py
``` 
Нужна е инсталация на допълнителни пакети.

[`data`]: https://github.com/nouuata/math4analysis/tree/main/data
[`keng`]: https://github.com/nouuata/math4analysis/tree/main/data/keng
[`matvs`]: https://github.com/nouuata/math4analysis/tree/main/data/matvs
[`olymp`]: https://github.com/nouuata/math4analysis/tree/main/data/olymp
[`pms`]: https://github.com/nouuata/math4analysis/tree/main/data/pms
[`merged`]: https://github.com/nouuata/math4analysis/tree/main/data/merged
[`derived`]: https://github.com/nouuata/math4analysis/tree/main/data/derived
[`by_region`]: https://github.com/nouuata/math4analysis/tree/main/data/derived/by_region
[`cdf`]: https://github.com/nouuata/math4analysis/tree/main/data/derived/cdf
[`classes`]: https://github.com/nouuata/math4analysis/tree/main/data/derived/classes
[`plots`]: https://github.com/nouuata/math4analysis/tree/main/plots
[`article`]: https://github.com/nouuata/math4analysis/tree/main/plots/article
[`by_acceptance`]: https://github.com/nouuata/math4analysis/tree/main/plots/by_acceptance
[`plot_by_region`]: https://github.com/nouuata/math4analysis/tree/main/plots/by_region
[`plot_cdf`]: https://github.com/nouuata/math4analysis/tree/main/plots/cdf
[`hist`]: https://github.com/nouuata/math4analysis/tree/main/plots/hist
[`joint_3d`]: https://github.com/nouuata/math4analysis/tree/main/plots/joint_3d
[`joint_dist`]: https://github.com/nouuata/math4analysis/tree/main/plots/joint_dist
[`with_size`]: https://github.com/nouuata/math4analysis/tree/main/plots/joint_dist/with_size