from random import choice
from shapeworld import util
from shapeworld.captions import Attribute
from shapeworld.captioners import WorldCaptioner


class LabelAttributeCaptioner(WorldCaptioner):

    def __init__(
        self,
        label,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=1.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0
    ):
        super(LabelAttributeCaptioner, self).__init__(
            internal_captioners=(),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.label = label

    def set_realizer(self, realizer):
        if not super(LabelAttributeCaptioner, self).set_realizer(realizer):
            return False

        self.shapes = list(realizer.attributes.get('shape', ()))
        self.colors = list(realizer.attributes.get('color', ()))
        self.textures = list(realizer.attributes.get('texture', ()))
        assert self.shapes or self.colors or self.textures

        return True

    def rpn_length(self):
        return 1

    def rpn_symbols(self):
        return super(LabelAttributeCaptioner, self).rpn_symbols() | \
            {'{}-{}-{}'.format(Attribute.__name__, 'shape', value) for value in self.shapes} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'color', value) for value in self.colors} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'texture', value) for value in self.textures}

    def incorrect_possible(self):
        return True

    def caption(self, predication, world):
        if predication.num_agreeing == 0:
            return None

        entity = choice(list(set((entity.shape.name, entity.color.name, entity.texture.name) for entity in predication.agreeing)))
        # predication.random_agreeing_entity()

        if world.meta[self.label] == 'shape':
            attribute = Attribute(predtype='shape', value=entity[0])

        elif world.meta[self.label] == 'color':
            attribute = Attribute(predtype='color', value=entity[1])

        elif world.meta[self.label] == 'texture':
            attribute = Attribute(predtype='texture', value=entity[2])

        if predication.contradictory(predicate=attribute):
            assert False
        elif not self.pragmatical_redundancy and predication.num_entities > 1 and predication.redundant(predicate=attribute):
            assert False
            return None

        attribute.apply_to_predication(predication=predication)

        return attribute

    def incorrect(self, caption, predication, world):
        if world.meta[self.label] == 'shape':  # random (existing) shape
            values = list(set(entity.shape.name for entity in world.entities if entity.shape.name in self.shapes and entity.shape.name != caption.value))

        elif world.meta[self.label] == 'color':  # random (existing) color
            values = list(set(entity.color.name for entity in world.entities if entity.color.name in self.colors and entity.color.name != caption.value))

        elif world.meta[self.label] == 'texture':  # random (existing) texture
            values = list(set(entity.texture.name for entity in world.entities if entity.texture.name in self.textures and entity.texture.name != caption.value))

        assert len(values) >= 1
        caption.value = choice(values)

        caption.apply_to_predication(predication=predication)

        return True
