import unittest
import numpy as np

from tfm_interpretability import (
    eigen_activation_map, grid_boxes, heatmap_mass_inside, mask_tiles,
    minmax, pointing_game, target_score, unletterbox_map,
)


class InterpretabilityTests(unittest.TestCase):
    def test_minmax_constant_is_zero(self):
        self.assertTrue(np.array_equal(minmax(np.ones((2,2))),np.zeros((2,2))))

    def test_eigen_map_shape_and_range(self):
        rng=np.random.default_rng(42); activation=rng.normal(size=(8,6,5))
        result=eigen_activation_map(activation)
        self.assertEqual(result.shape,(6,5)); self.assertGreaterEqual(result.min(),0); self.assertLessEqual(result.max(),1)

    def test_unletterbox_returns_original_shape(self):
        heat=np.arange(24*24,dtype=float).reshape(24,24)
        result=unletterbox_map(heat,(720,1280),768)
        self.assertEqual(result.shape,(720,1280)); self.assertTrue(np.isfinite(result).all())

    def test_grid_covers_image_once(self):
        boxes=grid_boxes(11,17,6); coverage=np.zeros((11,17),dtype=int)
        for x1,y1,x2,y2 in boxes: coverage[y1:y2,x1:x2]+=1
        self.assertEqual(len(boxes),36); self.assertTrue((coverage==1).all())

    def test_mask_tiles_uses_blurred_region_only(self):
        image=np.zeros((6,6,3),dtype=np.uint8); blurred=np.full_like(image,9); boxes=grid_boxes(6,6,2)
        result=mask_tiles(image,blurred,boxes,[0])
        self.assertTrue((result[:3,:3]==9).all()); self.assertTrue((result[3:,3:]==0).all())

    def test_target_score_requires_class_and_overlap(self):
        preds=[{"class_id":0,"confidence":.8,"xyxy":[.1,.1,.5,.5]},
               {"class_id":1,"confidence":.9,"xyxy":[.1,.1,.5,.5]}]
        self.assertEqual(target_score(preds,0,[.1,.1,.5,.5],.3),.8)
        self.assertEqual(target_score(preds,0,[.7,.7,.9,.9],.3),0)

    def test_target_score_prioritizes_identity_over_confidence(self):
        preds=[{"class_id":0,"confidence":.4,"xyxy":[.1,.1,.5,.5]},
               {"class_id":0,"confidence":.9,"xyxy":[.2,.2,.6,.6]}]
        self.assertEqual(target_score(preds,0,[.1,.1,.5,.5],.3),.4)

    def test_localization_metrics(self):
        heat=np.zeros((4,4)); heat[1,1]=2; box=[.25,.25,.5,.5]
        self.assertTrue(pointing_game(heat,box)); self.assertAlmostEqual(heatmap_mass_inside(heat,box),1)


if __name__=="__main__": unittest.main()
