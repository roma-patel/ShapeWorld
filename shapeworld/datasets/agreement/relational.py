from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import ReinforcedAttributesGenerator
from shapeworld.captioners import RegularTypeCaptioner, UniqueTypeCaptioner, RelationCaptioner, NegationRelationCaptioner, ExistentialCaptioner


class RelationalDataset(CaptionAgreementDataset):

    def __init__(
        self,
        world_size=64,
        world_color='black',
        shapes=('square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'),
        colors=('red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray'),
        textures=('solid',),
        rotation=True,
        size_range=(0.1, 0.25),
        distortion_range=(2.0, 3.0),
        shade_range=0.4,
        collision_tolerance=0.25,
        collision_shade_difference=0.5,
        boundary_tolerance=None,
        entity_counts=(3, 4, 5, 6, 7, 8, 9, 10),
        train_entity_counts=None,
        validation_entity_counts=None,
        test_entity_counts=None,
        validation_count_rate=0.5,
        test_count_rate=0.5,
        validation_combinations=None,
        test_combinations=None,
        validation_space_rate_range=(0.0, 1.0),
        test_space_rate_range=(0.0, 1.0),
        validation_combination_rate=0.5,
        test_combination_rate=0.5,
        max_provoke_collision_rate=0.33,
        relations=None,
        negation=True,
        caption_size=15,
        vocabulary=('.', 'a', 'above', 'an', 'as', 'behind', 'below', 'bigger', 'blue', 'circle', 'closer', 'color', 'cross', 'cyan', 'darker', 'different', 'ellipse', 'farther', 'from', 'front', 'gray', 'green', 'in', 'is', 'left', 'lighter', 'magenta', 'not', 'of', 'pentagon', 'rectangle', 'red', 'right', 'same', 'semicircle', 'shape', 'smaller', 'square', 'than', 'the', 'to', 'triangle', 'yellow'),
        correct_ratio=0.5,
        train_correct_ratio=None,
        validation_correct_ratio=None,
        test_correct_ratio=None,
        worlds_per_instance=1,
        captions_per_instance=1,
        pixel_noise_stddev=None,
        caption_realizer='dmrs',
        language=None
    ):

        world_generator = ReinforcedAttributesGenerator(
            world_size=world_size,
            world_color=world_color,
            shapes=shapes,
            colors=colors,
            textures=textures,
            rotation=rotation,
            size_range=size_range,
            distortion_range=distortion_range,
            shade_range=shade_range,
            collision_tolerance=collision_tolerance,
            collision_shade_difference=collision_shade_difference,
            boundary_tolerance=boundary_tolerance,
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            validation_count_rate=validation_count_rate,
            test_entity_counts=test_entity_counts,
            test_count_rate=test_count_rate,
            validation_combinations=validation_combinations,
            validation_space_rate_range=validation_space_rate_range,
            validation_combination_rate=validation_combination_rate,
            test_combinations=test_combinations,
            test_space_rate_range=test_space_rate_range,
            test_combination_rate=test_combination_rate,
            max_provoke_collision_rate=max_provoke_collision_rate,
            reinforcement_range=(1, 1)
        )

        relation_captioner = RelationCaptioner(
            reference_captioner=RegularTypeCaptioner(),
            comparison_captioner=UniqueTypeCaptioner(),
            relations=relations
        )
        if negation:
            relation_captioner = NegationRelationCaptioner(
                relation_captioner=relation_captioner
            )

        world_captioner = ExistentialCaptioner(
            restrictor_captioner=RegularTypeCaptioner(),
            body_captioner=relation_captioner
        )

        super(RelationalDataset, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            correct_ratio=correct_ratio,
            train_correct_ratio=train_correct_ratio,
            validation_correct_ratio=validation_correct_ratio,
            test_correct_ratio=test_correct_ratio,
            worlds_per_instance=worlds_per_instance,
            captions_per_instance=captions_per_instance,
            pixel_noise_stddev=pixel_noise_stddev,
            caption_realizer=caption_realizer,
            language=language
        )
