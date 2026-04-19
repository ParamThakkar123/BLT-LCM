# Error Analysis: Phase 2 (BLT) vs Phase 1 Tokenizers

**Sentences analyzed:** 50000
**Total regressions (BLT fertility > Phase1 best):** 487
**Top 100 worst regressions analyzed below.**

## Category Distribution

| Category | Count | % of 100 |
|----------|------:|--------:|
| Rare Unicode / uncommon Devanagari | 64 | 64% |
| Very short sentence (< 5 words) | 37 | 37% |
| Long compound words | 11 | 11% |
| Other | 8 | 8% |
| Domain-specific: medical | 2 | 2% |
| Domain-specific: legal | 1 | 1% |

## Summary Statistics

- Average fertility delta (top 100): **0.3753**
- Worst regression delta: **2.0**
- 100th regression delta: **0.1765**

## Top 100 Failure Sentences

### 1. (delta=2.0000)
**Categories:** Very short sentence (< 5 words)

> आठवतंय

| Metric | Value |
|--------|------:|
| Words | 1 |
| Phase1 Augmented fertility | 2.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 3.0 |
| BLT patches | 3 |
| Delta (regression) | 2.0 |

### 2. (delta=1.0000)
**Categories:** Rare Unicode / uncommon Devanagari, Very short sentence (< 5 words)

> त्यांनी माझ्यावर दादागिरी केली

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 2.0 |
| BLT patches | 8 |
| Delta (regression) | 1.0 |

### 3. (delta=0.8000)
**Categories:** Rare Unicode / uncommon Devanagari

> स्वतःला वाचवण्यासाठी काहीच वेळ नव्हता

| Metric | Value |
|--------|------:|
| Words | 5 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.8 |
| BLT patches | 9 |
| Delta (regression) | 0.8 |

### 4. (delta=0.7500)
**Categories:** Very short sentence (< 5 words)

> तिचं उत्तर आठवते आहे

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.25 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.75 |
| BLT patches | 7 |
| Delta (regression) | 0.75 |

### 5. (delta=0.7500)
**Categories:** Rare Unicode / uncommon Devanagari, Very short sentence (< 5 words)

> स्वच्छ पाण्याचे दुर्भिक्ष्य आहे

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.75 |
| BLT patches | 7 |
| Delta (regression) | 0.75 |

### 6. (delta=0.7500)
**Categories:** Very short sentence (< 5 words)

> मीही तुझ्यासोबत असेन लवकरच

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.5 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.75 |
| BLT patches | 7 |
| Delta (regression) | 0.75 |

### 7. (delta=0.6667)
**Categories:** Very short sentence (< 5 words)

> नेमकं हेच घडलं

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.6667 |
| BLT patches | 5 |
| Delta (regression) | 0.6667 |

### 8. (delta=0.6667)
**Categories:** Very short sentence (< 5 words)

> प्रेम करून पहा

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.6667 |
| BLT patches | 5 |
| Delta (regression) | 0.6667 |

### 9. (delta=0.6667)
**Categories:** Very short sentence (< 5 words)

> फारच सुमार सामग्री

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.6667 |
| BLT patches | 5 |
| Delta (regression) | 0.6667 |

### 10. (delta=0.6667)
**Categories:** Very short sentence (< 5 words)

> का तिने विचारले

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.6667 |
| BLT patches | 5 |
| Delta (regression) | 0.6667 |

### 11. (delta=0.6667)
**Categories:** Very short sentence (< 5 words)

> मी तुझ्यासोबत असेन

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 1.6667 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.6667 |
| BLT patches | 5 |
| Delta (regression) | 0.6667 |

### 12. (delta=0.6667)
**Categories:** Very short sentence (< 5 words)

> आम्ही चुरशीने लढलो

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.6667 |
| BLT patches | 5 |
| Delta (regression) | 0.6667 |

### 13. (delta=0.6000)
**Categories:** Other

> त्यांनी विचलित होता कामा नये

| Metric | Value |
|--------|------:|
| Words | 5 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.6 |
| BLT patches | 8 |
| Delta (regression) | 0.6 |

### 14. (delta=0.5000)
**Categories:** Very short sentence (< 5 words)

> आज यातील काहीच नाही

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 6 |
| Delta (regression) | 0.5 |

### 15. (delta=0.5000)
**Categories:** Very short sentence (< 5 words)

> माझा चावा घेतला आहे

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 6 |
| Delta (regression) | 0.5 |

### 16. (delta=0.5000)
**Categories:** Very short sentence (< 5 words)

> सुनावणीनंतर उत्सुकता शिगेला पोहोचली

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 6 |
| Delta (regression) | 0.5 |

### 17. (delta=0.5000)
**Categories:** Very short sentence (< 5 words)

> त्याचे अनेक हिरोज होते

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.25 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.5 |
| BLT patches | 6 |
| Delta (regression) | 0.5 |

### 18. (delta=0.5000)
**Categories:** Rare Unicode / uncommon Devanagari

> त्याला प्राप्त करणे आमचे कर्तव्य आहे”

| Metric | Value |
|--------|------:|
| Words | 6 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.1667 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 9 |
| Delta (regression) | 0.5 |

### 19. (delta=0.5000)
**Categories:** Rare Unicode / uncommon Devanagari

> परंतु आम्ही त्यावर काम करत आहोत

| Metric | Value |
|--------|------:|
| Words | 6 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 9 |
| Delta (regression) | 0.5 |

### 20. (delta=0.5000)
**Categories:** Rare Unicode / uncommon Devanagari

> परंतु ते तसे नव्हते तर ती एक सरकारी प्रकरण होती

| Metric | Value |
|--------|------:|
| Words | 10 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 15 |
| Delta (regression) | 0.5 |

### 21. (delta=0.5000)
**Categories:** Rare Unicode / uncommon Devanagari, Very short sentence (< 5 words)

> त्याच्यामध्ये भरपूर आत्मविश्वास आलाय

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 6 |
| Delta (regression) | 0.5 |

### 22. (delta=0.5000)
**Categories:** Rare Unicode / uncommon Devanagari, Very short sentence (< 5 words)

> वटवाघळे प्रत्यक्षात आंधळी नसतात

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 2.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.5 |
| BLT patches | 6 |
| Delta (regression) | 0.5 |

### 23. (delta=0.5000)
**Categories:** Long compound words, Rare Unicode / uncommon Devanagari, Very short sentence (< 5 words)

> प्रोत्साहनाच्या संदेशांसाठी आभारी आहे”

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.5 |
| Phase1 Retrained fertility | 1.75 |
| Phase1 Best | 1.5 (augmented) |
| BLT fertility | 2.0 |
| BLT patches | 8 |
| Delta (regression) | 0.5 |

### 24. (delta=0.5000)
**Categories:** Very short sentence (< 5 words)

> अभ्यास इतिहास

| Metric | Value |
|--------|------:|
| Words | 2 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 3 |
| Delta (regression) | 0.5 |

### 25. (delta=0.5000)
**Categories:** Very short sentence (< 5 words)

> ओह नाही

| Metric | Value |
|--------|------:|
| Words | 2 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 3 |
| Delta (regression) | 0.5 |

### 26. (delta=0.5000)
**Categories:** Very short sentence (< 5 words)

> शक्यच नाही

| Metric | Value |
|--------|------:|
| Words | 2 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 3 |
| Delta (regression) | 0.5 |

### 27. (delta=0.5000)
**Categories:** Very short sentence (< 5 words)

> यामुळे आमच्यात फूट पडली

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 6 |
| Delta (regression) | 0.5 |

### 28. (delta=0.4545)
**Categories:** Long compound words, Rare Unicode / uncommon Devanagari

> आणि कार्यालयांमध्ये पाहणाऱ्या लोकांची संख्या मोजण्यात नेल्सनला सामान्यतः काही समस्या येते

| Metric | Value |
|--------|------:|
| Words | 11 |
| Phase1 Augmented fertility | 1.0909 |
| Phase1 Retrained fertility | 1.1818 |
| Phase1 Best | 1.0909 (augmented) |
| BLT fertility | 1.5455 |
| BLT patches | 17 |
| Delta (regression) | 0.4545 |

### 29. (delta=0.4286)
**Categories:** Rare Unicode / uncommon Devanagari

> आणि याच पद्धतीने तो महत्वपूर्ण गोल झाला

| Metric | Value |
|--------|------:|
| Words | 7 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.4286 |
| BLT patches | 10 |
| Delta (regression) | 0.4286 |

### 30. (delta=0.4286)
**Categories:** Rare Unicode / uncommon Devanagari

> कोणी अशा प्रकारे एखाद्याच्या मागे लागते का

| Metric | Value |
|--------|------:|
| Words | 7 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.4286 |
| BLT patches | 10 |
| Delta (regression) | 0.4286 |

### 31. (delta=0.4000)
**Categories:** Other

> नाही नाही नाही नाही नाही

| Metric | Value |
|--------|------:|
| Words | 5 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.4 |
| BLT patches | 7 |
| Delta (regression) | 0.4 |

### 32. (delta=0.4000)
**Categories:** Rare Unicode / uncommon Devanagari, Domain-specific: medical

> त्याचा नंतर रुग्णालयात मृत्यू झाला

| Metric | Value |
|--------|------:|
| Words | 5 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.4 |
| BLT patches | 7 |
| Delta (regression) | 0.4 |

### 33. (delta=0.4000)
**Categories:** Rare Unicode / uncommon Devanagari

> स्वच्छ पाणी दुर्मीळ झाले आहेत

| Metric | Value |
|--------|------:|
| Words | 5 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.4 |
| BLT patches | 7 |
| Delta (regression) | 0.4 |

### 34. (delta=0.3750)
**Categories:** Rare Unicode / uncommon Devanagari

> काहीवेळेस यामुळे अल्प अंतरात तापमानामध्ये नाटकीय बदल होतात

| Metric | Value |
|--------|------:|
| Words | 8 |
| Phase1 Augmented fertility | 1.25 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.375 |
| BLT patches | 11 |
| Delta (regression) | 0.375 |

### 35. (delta=0.3750)
**Categories:** Rare Unicode / uncommon Devanagari

> यातील प्रत्येक खेळाडूने बाकी ठेवलेल्या गोष्टींबद्दल आता आहे

| Metric | Value |
|--------|------:|
| Words | 8 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.375 |
| BLT patches | 11 |
| Delta (regression) | 0.375 |

### 36. (delta=0.3750)
**Categories:** Rare Unicode / uncommon Devanagari

> “मला माझ्या देशासाठी एक खेळाडू म्हणून खेळायचे होते

| Metric | Value |
|--------|------:|
| Words | 8 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.125 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.375 |
| BLT patches | 11 |
| Delta (regression) | 0.375 |

### 37. (delta=0.3333)
**Categories:** Rare Unicode / uncommon Devanagari, Very short sentence (< 5 words)

> अप्रतिम लिविंगस्टनकडून विश्लेषण

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 2.0 |
| Phase1 Retrained fertility | 1.6667 |
| Phase1 Best | 1.6667 (retrained) |
| BLT fertility | 2.0 |
| BLT patches | 6 |
| Delta (regression) | 0.3333 |

### 38. (delta=0.3333)
**Categories:** Rare Unicode / uncommon Devanagari

> ते गेरार्डच्या नेतृत्वाखाली होते तितक्याच प्रमाणात ते त्यांच्या प्रमाणित खेळापेक्षा कमी पडले

| Metric | Value |
|--------|------:|
| Words | 12 |
| Phase1 Augmented fertility | 1.4167 |
| Phase1 Retrained fertility | 1.0833 |
| Phase1 Best | 1.0833 (retrained) |
| BLT fertility | 1.4167 |
| BLT patches | 17 |
| Delta (regression) | 0.3333 |

### 39. (delta=0.3333)
**Categories:** Rare Unicode / uncommon Devanagari

> या शहरामध्ये तुर्कीच्या बाहेर सर्वाधिक प्रमाणात तुर्कीश लोकसंख्या आहे

| Metric | Value |
|--------|------:|
| Words | 9 |
| Phase1 Augmented fertility | 1.2222 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.3333 |
| BLT patches | 12 |
| Delta (regression) | 0.3333 |

### 40. (delta=0.3333)
**Categories:** Very short sentence (< 5 words)

> ते आवडले नाही

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.3333 |
| BLT patches | 4 |
| Delta (regression) | 0.3333 |

### 41. (delta=0.3333)
**Categories:** Rare Unicode / uncommon Devanagari

> त्यांनी व्यासपीठाच्या मागे माझ्यावर दादागिरी केली

| Metric | Value |
|--------|------:|
| Words | 6 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.3333 |
| BLT patches | 8 |
| Delta (regression) | 0.3333 |

### 42. (delta=0.3333)
**Categories:** Long compound words, Rare Unicode / uncommon Devanagari

> त्यांनी आरोपकर्त्यांच्या हेतूंवर संशय घ्यायचा नसतो

| Metric | Value |
|--------|------:|
| Words | 6 |
| Phase1 Augmented fertility | 1.5 |
| Phase1 Retrained fertility | 1.3333 |
| Phase1 Best | 1.3333 (retrained) |
| BLT fertility | 1.6667 |
| BLT patches | 10 |
| Delta (regression) | 0.3333 |

### 43. (delta=0.3333)
**Categories:** Rare Unicode / uncommon Devanagari

> केवळ खेळच नाही तर त्यापेक्षाही तो अधिक काही देतो

| Metric | Value |
|--------|------:|
| Words | 9 |
| Phase1 Augmented fertility | 1.1111 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.3333 |
| BLT patches | 12 |
| Delta (regression) | 0.3333 |

### 44. (delta=0.3333)
**Categories:** Rare Unicode / uncommon Devanagari

> माझ्या नावाचा वापर करून ते स्वतःचा प्रचार करू इच्छितात

| Metric | Value |
|--------|------:|
| Words | 9 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.3333 |
| BLT patches | 12 |
| Delta (regression) | 0.3333 |

### 45. (delta=0.3333)
**Categories:** Very short sentence (< 5 words)

> ते सामान्य आहे

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.3333 |
| BLT patches | 4 |
| Delta (regression) | 0.3333 |

### 46. (delta=0.3333)
**Categories:** Very short sentence (< 5 words)

> इथून सुरुवात करा

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.3333 |
| BLT patches | 4 |
| Delta (regression) | 0.3333 |

### 47. (delta=0.3333)
**Categories:** Rare Unicode / uncommon Devanagari

> त्यांच्याकडे ही सुत्रे असती तर आम्ही आतापर्यंत बाहेर असतो असे ते म्हणाले

| Metric | Value |
|--------|------:|
| Words | 12 |
| Phase1 Augmented fertility | 1.3333 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.3333 |
| BLT patches | 16 |
| Delta (regression) | 0.3333 |

### 48. (delta=0.3333)
**Categories:** Very short sentence (< 5 words)

> मी केलेले नाही”

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 1.3333 |
| Phase1 Retrained fertility | 1.3333 |
| Phase1 Best | 1.3333 (augmented) |
| BLT fertility | 1.6667 |
| BLT patches | 5 |
| Delta (regression) | 0.3333 |

### 49. (delta=0.3333)
**Categories:** Very short sentence (< 5 words)

> ती रडली का

| Metric | Value |
|--------|------:|
| Words | 3 |
| Phase1 Augmented fertility | 1.3333 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.3333 |
| BLT patches | 4 |
| Delta (regression) | 0.3333 |

### 50. (delta=0.3077)
**Categories:** Rare Unicode / uncommon Devanagari

> माझे नाव घेऊन ते प्रसिद्धी मिळवू पाहत आहेत परंतु हा कामाचा भाग आहे

| Metric | Value |
|--------|------:|
| Words | 13 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.3077 |
| BLT patches | 17 |
| Delta (regression) | 0.3077 |

### 51. (delta=0.3000)
**Categories:** Rare Unicode / uncommon Devanagari

> हा माझा इतिहास आहे मग मी भविष्याबद्दल सकारात्मक का नसेन

| Metric | Value |
|--------|------:|
| Words | 10 |
| Phase1 Augmented fertility | 1.1 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.3 |
| BLT patches | 13 |
| Delta (regression) | 0.3 |

### 52. (delta=0.3000)
**Categories:** Rare Unicode / uncommon Devanagari, Domain-specific: medical

> सात जणांना रुग्णालयात नेण्यात आले आहे असे अधिकाऱ्यांनी शुक्रवारी सांगितले

| Metric | Value |
|--------|------:|
| Words | 10 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.3 |
| BLT patches | 13 |
| Delta (regression) | 0.3 |

### 53. (delta=0.2857)
**Categories:** Rare Unicode / uncommon Devanagari, Domain-specific: legal

> कायदेशीररित्या बंधनकारक नसले तरी संसदेच्या पुरेशा संख्येतील सदस्यांनी म्हटले आहे की हे मत निर्णायक होण्यासाठी आपण या मताच्या निकालाचे पालन करु

| Metric | Value |
|--------|------:|
| Words | 21 |
| Phase1 Augmented fertility | 1.0952 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2857 |
| BLT patches | 27 |
| Delta (regression) | 0.2857 |

### 54. (delta=0.2857)
**Categories:** Rare Unicode / uncommon Devanagari

> त्या सुरक्षित असतील अशी मी आशा करतो

| Metric | Value |
|--------|------:|
| Words | 7 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.2857 |
| BLT patches | 9 |
| Delta (regression) | 0.2857 |

### 55. (delta=0.2727)
**Categories:** Rare Unicode / uncommon Devanagari

> रोसा वादळाच्या प्रभावाने नैऋत्य अमेरिकेत व्यापक प्रमाणात जोरदार पाऊस पडणार आहे

| Metric | Value |
|--------|------:|
| Words | 11 |
| Phase1 Augmented fertility | 1.0909 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2727 |
| BLT patches | 14 |
| Delta (regression) | 0.2727 |

### 56. (delta=0.2727)
**Categories:** Long compound words, Rare Unicode / uncommon Devanagari

> स्वतंत्र शाळा त्यांच्या विद्यार्थ्यांना पुढील आणि उच्च शिक्षणासाठी त्यांच्या निवडलेल्या करियरसाठी आणि एक जागतिक नागरिक म्हणून त्यांच्या स्थानासाठी तयारी करण्याचे ध्येय राखतात

| Metric | Value |
|--------|------:|
| Words | 22 |
| Phase1 Augmented fertility | 1.4091 |
| Phase1 Retrained fertility | 1.0455 |
| Phase1 Best | 1.0455 (retrained) |
| BLT fertility | 1.3182 |
| BLT patches | 29 |
| Delta (regression) | 0.2727 |

### 57. (delta=0.2727)
**Categories:** Other

> गुरुवार सुरु होत आहे हि युरोपातील अजून एक मोठी रात्र आहे

| Metric | Value |
|--------|------:|
| Words | 11 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.2727 |
| BLT patches | 14 |
| Delta (regression) | 0.2727 |

### 58. (delta=0.2727)
**Categories:** Rare Unicode / uncommon Devanagari

> वायडे आपल्या मित्राला वाचवण्यासाठी मध्ये पडले आणि सिम्पसने त्यांच्यावर गोळी झाडली

| Metric | Value |
|--------|------:|
| Words | 11 |
| Phase1 Augmented fertility | 1.5455 |
| Phase1 Retrained fertility | 1.1818 |
| Phase1 Best | 1.1818 (retrained) |
| BLT fertility | 1.4545 |
| BLT patches | 16 |
| Delta (regression) | 0.2727 |

### 59. (delta=0.2500)
**Categories:** Other

> ती वारंवार म्हणत होती हे अतिशय वाईट आहे

| Metric | Value |
|--------|------:|
| Words | 8 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.25 |
| BLT patches | 10 |
| Delta (regression) | 0.25 |

### 60. (delta=0.2500)
**Categories:** Very short sentence (< 5 words)

> हे अतिशय वाईट आहे

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.25 |
| BLT patches | 5 |
| Delta (regression) | 0.25 |

### 61. (delta=0.2500)
**Categories:** Very short sentence (< 5 words)

> तसे अद्याप घडलेले नाही

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.25 |
| BLT patches | 5 |
| Delta (regression) | 0.25 |

### 62. (delta=0.2500)
**Categories:** Very short sentence (< 5 words)

> माणूस ते काय आहे

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.25 |
| BLT patches | 5 |
| Delta (regression) | 0.25 |

### 63. (delta=0.2500)
**Categories:** Rare Unicode / uncommon Devanagari

> अशा संबंधांतील अपुऱ्या संरक्षणामुळे रोगांचे बहुतेक संक्रमण होते

| Metric | Value |
|--------|------:|
| Words | 8 |
| Phase1 Augmented fertility | 1.375 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.25 |
| BLT patches | 10 |
| Delta (regression) | 0.25 |

### 64. (delta=0.2500)
**Categories:** Long compound words, Rare Unicode / uncommon Devanagari

> स्रोतांच्या म्हणण्यानुसार ही कल्पना पक्षाच्या देशाला कर्मचाऱ्यांच्या हिताकडे झुकवण्याच्या आर्थिक विषयपत्रिकेत आणि धोरणांमध्ये चपखल बसणारी आहे

| Metric | Value |
|--------|------:|
| Words | 16 |
| Phase1 Augmented fertility | 1.5 |
| Phase1 Retrained fertility | 1.0625 |
| Phase1 Best | 1.0625 (retrained) |
| BLT fertility | 1.3125 |
| BLT patches | 21 |
| Delta (regression) | 0.25 |

### 65. (delta=0.2500)
**Categories:** Rare Unicode / uncommon Devanagari, Very short sentence (< 5 words)

> हो नक्कीच केनेडी म्हणाले

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.25 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.25 |
| BLT patches | 5 |
| Delta (regression) | 0.25 |

### 66. (delta=0.2500)
**Categories:** Rare Unicode / uncommon Devanagari

> मी आज देखील कामाच्या तणाव आणि दबावाच्या परिस्थितींमधून जातो पण तरी देखील व्यवस्थापन माझ्यासाठी उचित वाटते

| Metric | Value |
|--------|------:|
| Words | 16 |
| Phase1 Augmented fertility | 1.125 |
| Phase1 Retrained fertility | 1.0625 |
| Phase1 Best | 1.0625 (retrained) |
| BLT fertility | 1.3125 |
| BLT patches | 21 |
| Delta (regression) | 0.25 |

### 67. (delta=0.2500)
**Categories:** Very short sentence (< 5 words)

> मी चालू शकत नव्हतो

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.25 |
| BLT patches | 5 |
| Delta (regression) | 0.25 |

### 68. (delta=0.2500)
**Categories:** Rare Unicode / uncommon Devanagari, Very short sentence (< 5 words)

> कुत्र्याची काळजी कोण घेणार

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.25 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.25 |
| BLT patches | 5 |
| Delta (regression) | 0.25 |

### 69. (delta=0.2500)
**Categories:** Very short sentence (< 5 words)

> पिडीताने हल्लेखोराला पहिले नाही

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.25 |
| Phase1 Retrained fertility | 1.25 |
| Phase1 Best | 1.25 (augmented) |
| BLT fertility | 1.5 |
| BLT patches | 6 |
| Delta (regression) | 0.25 |

### 70. (delta=0.2500)
**Categories:** Long compound words, Rare Unicode / uncommon Devanagari, Very short sentence (< 5 words)

> इंडोनेशियामध्ये भूकंप आणि त्सुनामी

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.25 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.25 |
| BLT patches | 5 |
| Delta (regression) | 0.25 |

### 71. (delta=0.2500)
**Categories:** Long compound words, Rare Unicode / uncommon Devanagari

> प्रसारमाध्यमांनी दिलेल्या अहवालात त्याला शनिवारी न्यायालयामध्ये उपस्थित केले जिथे त्याला विचारणा केली गेली असताना त्याने स्वतःला दोषी नसल्याचे घोषित केले

| Metric | Value |
|--------|------:|
| Words | 20 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.25 |
| BLT patches | 25 |
| Delta (regression) | 0.25 |

### 72. (delta=0.2500)
**Categories:** Very short sentence (< 5 words)

> हा सगळा बनाव आहे

| Metric | Value |
|--------|------:|
| Words | 4 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.25 |
| BLT patches | 5 |
| Delta (regression) | 0.25 |

### 73. (delta=0.2500)
**Categories:** Rare Unicode / uncommon Devanagari

> त्याने तिला आपल्या आयुष्यातून काढून टाकले हेल म्हणाले

| Metric | Value |
|--------|------:|
| Words | 8 |
| Phase1 Augmented fertility | 1.125 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.25 |
| BLT patches | 10 |
| Delta (regression) | 0.25 |

### 74. (delta=0.2308)
**Categories:** Rare Unicode / uncommon Devanagari

> आम्ही विशेषरित्या प्रशिक्षित सहकाऱ्यांमध्ये देखील गुंतवणूक केली आहे जे सल्ला देण्यासाठी तत्पर असतील

| Metric | Value |
|--------|------:|
| Words | 13 |
| Phase1 Augmented fertility | 1.2308 |
| Phase1 Retrained fertility | 1.0769 |
| Phase1 Best | 1.0769 (retrained) |
| BLT fertility | 1.3077 |
| BLT patches | 17 |
| Delta (regression) | 0.2308 |

### 75. (delta=0.2308)
**Categories:** Rare Unicode / uncommon Devanagari

> संघ चांगला आहे आणि मुले खरोखरच याचा आनंद घेत आहेत प्रशिक्षण चांगले आहे

| Metric | Value |
|--------|------:|
| Words | 13 |
| Phase1 Augmented fertility | 1.0769 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2308 |
| BLT patches | 16 |
| Delta (regression) | 0.2308 |

### 76. (delta=0.2308)
**Categories:** Rare Unicode / uncommon Devanagari

> २०१४ मध्ये टायलरने आत्महत्या केली त्यावेळी ११ वर्षांच्या असलेल्या भावानेच त्याला मृतावस्थेत बघितले

| Metric | Value |
|--------|------:|
| Words | 13 |
| Phase1 Augmented fertility | 1.1538 |
| Phase1 Retrained fertility | 1.0769 |
| Phase1 Best | 1.0769 (retrained) |
| BLT fertility | 1.3077 |
| BLT patches | 17 |
| Delta (regression) | 0.2308 |

### 77. (delta=0.2222)
**Categories:** Rare Unicode / uncommon Devanagari

> बार्स अदृश्य झाले हे समजण्यासारखे आहे असं ते म्हणाले

| Metric | Value |
|--------|------:|
| Words | 9 |
| Phase1 Augmented fertility | 1.4444 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2222 |
| BLT patches | 11 |
| Delta (regression) | 0.2222 |

### 78. (delta=0.2222)
**Categories:** Rare Unicode / uncommon Devanagari

> बाहेर पडण्याच्या तिच्या निर्णयाने समाविष्ट प्रत्येक जण अचंबित झाला

| Metric | Value |
|--------|------:|
| Words | 9 |
| Phase1 Augmented fertility | 1.1111 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2222 |
| BLT patches | 11 |
| Delta (regression) | 0.2222 |

### 79. (delta=0.2222)
**Categories:** Rare Unicode / uncommon Devanagari

> आणि अधिकाऱ्यांच्या म्हणण्यानुसार हे हाताळण्याचे मार्ग व साधने आहेत

| Metric | Value |
|--------|------:|
| Words | 9 |
| Phase1 Augmented fertility | 1.3333 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2222 |
| BLT patches | 11 |
| Delta (regression) | 0.2222 |

### 80. (delta=0.2222)
**Categories:** Other

> तिचे आणि माझ्या मुलीचे काय झाले मला माहीत नाही

| Metric | Value |
|--------|------:|
| Words | 9 |
| Phase1 Augmented fertility | 1.1111 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2222 |
| BLT patches | 11 |
| Delta (regression) | 0.2222 |

### 81. (delta=0.2222)
**Categories:** Rare Unicode / uncommon Devanagari

> परंतु तेथे रविवारच्या सकाळचे सहा वाजले होते आणि आम्ही आमच्या रविवारपर्यंत म्हणजेच त्यांच्या सोमवारपर्यंत ते करू शकणार नव्हतो

| Metric | Value |
|--------|------:|
| Words | 18 |
| Phase1 Augmented fertility | 1.1111 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2222 |
| BLT patches | 22 |
| Delta (regression) | 0.2222 |

### 82. (delta=0.2222)
**Categories:** Rare Unicode / uncommon Devanagari

> त्याला माझ्या प्रेमाची मी त्याला समजून घेण्याची गरज होती

| Metric | Value |
|--------|------:|
| Words | 9 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.2222 |
| BLT patches | 11 |
| Delta (regression) | 0.2222 |

### 83. (delta=0.2222)
**Categories:** Rare Unicode / uncommon Devanagari

> मला माहीत होते त्याचे तिच्यावर किती प्रेम होते ते

| Metric | Value |
|--------|------:|
| Words | 9 |
| Phase1 Augmented fertility | 1.1111 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2222 |
| BLT patches | 11 |
| Delta (regression) | 0.2222 |

### 84. (delta=0.2222)
**Categories:** Rare Unicode / uncommon Devanagari

> दुखातील स्त्रिया पुलावरून होणाऱ्या आत्महत्या टाळण्यासाठी कार्ड्स पोस्ट करतात

| Metric | Value |
|--------|------:|
| Words | 9 |
| Phase1 Augmented fertility | 1.4444 |
| Phase1 Retrained fertility | 1.1111 |
| Phase1 Best | 1.1111 (retrained) |
| BLT fertility | 1.3333 |
| BLT patches | 12 |
| Delta (regression) | 0.2222 |

### 85. (delta=0.2083)
**Categories:** Long compound words, Rare Unicode / uncommon Devanagari

> हे राष्ट्रपती त्यांच्या अगोदरच्या कोणत्याही राष्ट्रपतींपेक्षा जास्त प्रश्नोत्तरांच्या सत्रांचे आयोजन करतात त्या म्हणाल्या आणि नंतर पुरावे न देता म्हणाल्या आम्ही ते आकडे पाहिले आहेत

| Metric | Value |
|--------|------:|
| Words | 24 |
| Phase1 Augmented fertility | 1.375 |
| Phase1 Retrained fertility | 1.0417 |
| Phase1 Best | 1.0417 (retrained) |
| BLT fertility | 1.25 |
| BLT patches | 30 |
| Delta (regression) | 0.2083 |

### 86. (delta=0.2069)
**Categories:** Long compound words, Rare Unicode / uncommon Devanagari

> या वर्षी पहिल्यांदा प्रस्तुत केलेला हा अभ्यासक्रम विद्यार्थ्यांना चांगल्या झोपेची सवय कशा रीतीने विद्यार्थ्यांना आणि खेळाडूंना त्यांच्या कामगिरीसाठी आणि तसेच त्याच्या सामान्य आरोग्यास सुधारण्यासाठी जर...

| Metric | Value |
|--------|------:|
| Words | 29 |
| Phase1 Augmented fertility | 1.2069 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2069 |
| BLT patches | 35 |
| Delta (regression) | 0.2069 |

### 87. (delta=0.2000)
**Categories:** Other

> लक्षावधी लोक मारले गेले असते

| Metric | Value |
|--------|------:|
| Words | 5 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.2 |
| BLT patches | 6 |
| Delta (regression) | 0.2 |

### 88. (delta=0.2000)
**Categories:** Rare Unicode / uncommon Devanagari

> पुढील काही दिवसांमध्ये उन्हाळ्याचा रेंगाळलेला उष्मा पुन्हा वाढेल आणि विस्तारेल

| Metric | Value |
|--------|------:|
| Words | 10 |
| Phase1 Augmented fertility | 1.3 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2 |
| BLT patches | 12 |
| Delta (regression) | 0.2 |

### 89. (delta=0.2000)
**Categories:** Rare Unicode / uncommon Devanagari

> हा संघर्ष अधिकाधिक तणावपूर्ण होत चालला आहे असं त्यांनी सांगितलं

| Metric | Value |
|--------|------:|
| Words | 10 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.2 |
| BLT patches | 12 |
| Delta (regression) | 0.2 |

### 90. (delta=0.2000)
**Categories:** Rare Unicode / uncommon Devanagari

> मला वाटते हा आपल्या काळातील मोठ्या प्रश्नांपैकी खरोखरच एक प्रश्न आहे आपण ते कसे बदलावे

| Metric | Value |
|--------|------:|
| Words | 15 |
| Phase1 Augmented fertility | 1.2 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2 |
| BLT patches | 18 |
| Delta (regression) | 0.2 |

### 91. (delta=0.2000)
**Categories:** Rare Unicode / uncommon Devanagari

> आम्ही त्यांची परवानगी मागत नाही

| Metric | Value |
|--------|------:|
| Words | 5 |
| Phase1 Augmented fertility | 1.8 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2 |
| BLT patches | 6 |
| Delta (regression) | 0.2 |

### 92. (delta=0.2000)
**Categories:** Rare Unicode / uncommon Devanagari

> जेव्हा तो वेग एका दिशेने जात असतो तेव्हा तो मधल्या सामन्यांवर भरपूर दबाव निर्माण करतो

| Metric | Value |
|--------|------:|
| Words | 15 |
| Phase1 Augmented fertility | 1.0667 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2 |
| BLT patches | 18 |
| Delta (regression) | 0.2 |

### 93. (delta=0.2000)
**Categories:** Other

> परंतु खेळणे कठीण असू शकते

| Metric | Value |
|--------|------:|
| Words | 5 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.2 |
| BLT patches | 6 |
| Delta (regression) | 0.2 |

### 94. (delta=0.2000)
**Categories:** Other

> आणि मी तिला निरोप दिला

| Metric | Value |
|--------|------:|
| Words | 5 |
| Phase1 Augmented fertility | 1.0 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (augmented) |
| BLT fertility | 1.2 |
| BLT patches | 6 |
| Delta (regression) | 0.2 |

### 95. (delta=0.2000)
**Categories:** Rare Unicode / uncommon Devanagari

> कोणालाच माहीत नव्हते त्या म्हणाल्या

| Metric | Value |
|--------|------:|
| Words | 5 |
| Phase1 Augmented fertility | 1.2 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.2 |
| BLT patches | 6 |
| Delta (regression) | 0.2 |

### 96. (delta=0.1818)
**Categories:** Rare Unicode / uncommon Devanagari

> संविधानाच्या आधारे कायदे होतात व कायद्याच्या आधारे देश चालतो या कार्यक्रमात गृहराज्यमंत्री किरेन रिजीजू यांचेही भाषण झाले प्रत्येक जिल्ह्यात महिला आयोगाचं कार्यालय स्थापन करणार असल्याचं त्यांनी सांगित...

| Metric | Value |
|--------|------:|
| Words | 33 |
| Phase1 Augmented fertility | 1.1212 |
| Phase1 Retrained fertility | 1.0 |
| Phase1 Best | 1.0 (retrained) |
| BLT fertility | 1.1818 |
| BLT patches | 39 |
| Delta (regression) | 0.1818 |

### 97. (delta=0.1818)
**Categories:** Rare Unicode / uncommon Devanagari

> पुढे येत असलेल्या फुगवट्याने दक्षिण कॅलिफोर्निया किनारपट्टीच्या भागांमध्ये जोरदार पाऊस येईल

| Metric | Value |
|--------|------:|
| Words | 11 |
| Phase1 Augmented fertility | 1.2727 |
| Phase1 Retrained fertility | 1.0909 |
| Phase1 Best | 1.0909 (retrained) |
| BLT fertility | 1.2727 |
| BLT patches | 14 |
| Delta (regression) | 0.1818 |

### 98. (delta=0.1818)
**Categories:** Rare Unicode / uncommon Devanagari

> त्यांना सांगण्यात आले इतक्या सगळ्या महिला शक्तिहीन असताना तुमच्याकडे सत्ता आहे

| Metric | Value |
|--------|------:|
| Words | 11 |
| Phase1 Augmented fertility | 1.0909 |
| Phase1 Retrained fertility | 1.0909 |
| Phase1 Best | 1.0909 (augmented) |
| BLT fertility | 1.2727 |
| BLT patches | 14 |
| Delta (regression) | 0.1818 |

### 99. (delta=0.1818)
**Categories:** Long compound words, Rare Unicode / uncommon Devanagari

> मोरालेसच्या आक्रमक वक्तृत्वकौशल्याने उरलेल्या चिलीच्या प्रतिष्ठेस संपवून टाकले आहे त्यांनी सुचविले

| Metric | Value |
|--------|------:|
| Words | 11 |
| Phase1 Augmented fertility | 1.4545 |
| Phase1 Retrained fertility | 1.2727 |
| Phase1 Best | 1.2727 (retrained) |
| BLT fertility | 1.4545 |
| BLT patches | 16 |
| Delta (regression) | 0.1818 |

### 100. (delta=0.1765)
**Categories:** Long compound words, Rare Unicode / uncommon Devanagari

> देशभरातील अभ्यासात जवळजवळ निम्म्या लोकांनी सांगितले की ते त्यांच्या पतीपत्नी बरोबर दीर्घकाल काळजी घेण्याबद्दलच्या खर्चाविषयी बोलत होते

| Metric | Value |
|--------|------:|
| Words | 17 |
| Phase1 Augmented fertility | 1.4118 |
| Phase1 Retrained fertility | 1.0588 |
| Phase1 Best | 1.0588 (retrained) |
| BLT fertility | 1.2353 |
| BLT patches | 21 |
| Delta (regression) | 0.1765 |

