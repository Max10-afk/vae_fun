class Dense3dLayer(tf.keras.layers.Layer):
  """A dense layer with 3D kernel."""

  def __init__(self,
               num_attention_heads,
               size_per_head,
               initializer,
               activation,
               name=None,
               head_first=False,
               use_bias=True):
    """Constructor for dense layer with 3D kernel.

    Args:
      num_attention_heads: The size of output dimension.
      size_per_head: The size per attention head.
      initializer: Kernel initializer.
      activation: Actication function.
      name: The name scope of this layer.
      head_first: Whether to output head dimension before or after sequence dim.
      use_bias: Whether the layer uses a bias vector.
    """
    super(Dense3dLayer, self).__init__(name=name)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.initializer = initializer
    self.activation = activation
    self.head_first = head_first
    self.use_bias = use_bias

    with tf.compat.v1.variable_scope(name):
      hidden_size = self.num_attention_heads * self.size_per_head
      self.w = tf.compat.v1.get_variable(
          name="kernel",
          shape=[hidden_size, hidden_size],
          initializer=self.initializer)

      if self.use_bias:
        self.b = tf.compat.v1.get_variable(
            name="bias",
            shape=[hidden_size],
            initializer=tf.zeros_initializer())
      else:
        self.b = None

  def call(self, input_tensor):
    """Constructor for dense layer with 3D kernel.

    Args:
      input_tensor: float Tensor of shape [batch, seq_length, hidden_size].

    Returns:
      float logits Tensor.
    """
    hidden_size = self.num_attention_heads * self.size_per_head
    reshape_w = tf.reshape(
        self.w, [hidden_size, self.num_attention_heads, self.size_per_head])
    if self.head_first:
      ret = tf.einsum("abc,cde->adbe", input_tensor, reshape_w)
    else:
      ret = tf.einsum("abc,cde->abde", input_tensor, reshape_w)

    if self.use_bias:
      if self.head_first:
        reshape_b = tf.reshape(
            self.b, [1, self.num_attention_heads, 1, self.size_per_head])
      else:
        reshape_b = tf.reshape(
            self.b, [self.num_attention_heads, self.size_per_head])
      ret += reshape_b

    if self.activation is not None:
      return self.activation(ret)
    else:
      return ret


class Dense3dProjLayer(tf.keras.layers.Layer):
  """A dense layer with 3D kernel for projection."""

  def __init__(self,
               num_attention_heads,
               size_per_head,
               initializer,
               activation,
               name=None,
               use_bias=True):
    """Constructor for dense layer with 3D kernel for projection.

    Args:
      num_attention_heads: The size of output dimension.
      size_per_head: The size per attention head.
      initializer: Kernel initializer.
      activation: Actication function.
      name: The name scope of this layer.
      use_bias: Whether the layer uses a bias vector.
    """
    super(Dense3dProjLayer, self).__init__(name=name)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.initializer = initializer
    self.activation = activation
    self.use_bias = use_bias

    with tf.compat.v1.variable_scope(name):
      hidden_size = self.num_attention_heads * self.size_per_head
      self.w = tf.compat.v1.get_variable(
          name="kernel",
          shape=[hidden_size, hidden_size],
          initializer=self.initializer)

      if self.use_bias:
        self.b = tf.compat.v1.get_variable(
            name="bias",
            shape=[hidden_size],
            initializer=tf.zeros_initializer())
      else:
        self.b = None

  def call(self, input_tensor):
    """Constructor for dense layer with 3D kernel for projection.

    Args:
      input_tensor: float Tensor of shape [batch,from_seq_length,
        num_attention_heads, size_per_head].

    Returns:
      float logits Tensor.
    """
    hidden_size = self.num_attention_heads * self.size_per_head
    reshape_w = tf.reshape(
        self.w, [self.num_attention_heads, self.size_per_head, hidden_size])
    ret = tf.einsum("BFNH,NHD->BFD", input_tensor, reshape_w)

    if self.use_bias:
      ret += self.b

    if self.activation is not None:
      return self.activation(ret)
    else:
      return ret


class Dense2dLayer(tf.keras.layers.Layer):
  """A dense layer with 2D kernel."""

  def __init__(self,
               input_size,
               output_size,
               initializer,
               activation,
               name=None,
               use_bias=True):
    """Constructor for dense layer with 2D kernel.

    Args:
      input_size: The size of input dimension.
      output_size: The size of output dimension.
      initializer: Kernel initializer.
      activation: Actication function.
      name: The name scope of this layer.
      use_bias: Whether the layer uses a bias vector.
    """
    super(Dense2dLayer, self).__init__(name=name)
    self.input_size = input_size
    self.output_size = output_size
    self.initializer = initializer
    self.activation = activation
    self.use_bias = use_bias

    with tf.compat.v1.variable_scope(name):
      self.w = tf.compat.v1.get_variable(
          name="kernel",
          shape=[self.input_size, self.output_size],
          initializer=self.initializer)

      if self.use_bias:
        self.b = tf.compat.v1.get_variable(
            name="bias",
            shape=[self.output_size],
            initializer=tf.zeros_initializer())
      else:
        self.b = None

  def call(self, input_tensor):
    """Forward pass for dense layer with 2D kernel.

    Args:
      input_tensor: Float tensor with rank 3.

    Returns:
      float logits Tensor.
    """
    ret = tf.einsum("acb,cd->abd", input_tensor, self.w)

    if self.use_bias:
      ret += self.b

    if self.activation is not None:
      return self.activation(ret)
    else:
      return ret


class Dense2dLayer_mod(tf.keras.layers.Layer):
  """A dense layer with 2D kernel."""

  def __init__(self,
               input_size,
               output_size,
               initializer,
               activation,
               name=None,
               use_bias=True):
    """Constructor for dense layer with 2D kernel.

    Args:
      input_size: The size of input dimension.
      output_size: The size of output dimension.
      initializer: Kernel initializer.
      activation: Actication function.
      name: The name scope of this layer.
      use_bias: Whether the layer uses a bias vector.
    """
    super(Dense2dLayer_mod, self).__init__(name=name)
    self.input_size = input_size
    self.output_size = output_size
    self.initializer = initializer
    self.activation = activation
    self.use_bias = use_bias

    with tf.compat.v1.variable_scope(name):
      self.w = tf.compat.v1.get_variable(
          name="kernel",
          shape=[self.input_size, self.output_size],
          initializer=self.initializer)

      if self.use_bias:
        self.b = tf.compat.v1.get_variable(
            name="bias",
            shape=[self.output_size],
            initializer=tf.zeros_initializer())
      else:
        self.b = None

  def call(self, input_tensor):
    """Forward pass for dense layer with 2D kernel.

    Args:
      input_tensor: Float tensor with rank 3.

    Returns:
      float logits Tensor.
    """
    ret = tf.einsum("acb,cd->acd", input_tensor, self.w)

    if self.use_bias:
      ret += self.b

    if self.activation is not None:
      return self.activation(ret)
    else:
      return ret

