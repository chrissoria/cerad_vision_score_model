# Changelog

All notable changes to PraxisNet models are documented here.

## CircleScore

### v1.1 (2025-01-30)

**Training Data:** 222 images (was 220)

**Changes:**
- Added 2 additional `not_closed` training examples

**Performance Impact:**
| Metric | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| Circularity Specificity | 33% | 75% | +42% |
| Circularity F1 | 82% | 91% | +9% |
| Circularity Errors | 10 | 5 | -50% |

**Key Finding:** Small additions to minority classes can significantly improve specificity.

---

### v1.0 (2025-01-26)

**Training Data:** 220 images

**Initial Release Metrics:**
| Dimension | Precision | Recall | Specificity | F1 |
|-----------|-----------|--------|-------------|-----|
| Presence | 100% | 100% | 100% | 100% |
| Closure | 97% | 100% | 75% | 99% |
| Circularity | 74% | 92% | 33% | 82% |

**Notes:**
- First trained model for CERAD circle scoring
- DINOv2 ViT-B/14 backbone with LP-FT training strategy
- Circularity identified as weakest dimension due to subjective boundaries
